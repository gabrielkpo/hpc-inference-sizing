"""Vast.ai driver: find an offer, rent it, guard it, destroy it (protocol §10).

Safety rules, enforced here and not left to the operator:

- The API key comes from `VAST_API_KEY` in the environment only. It is never printed, logged,
  written to a file or sent anywhere but the Vast.ai API.
- Renting needs `confirm=True` (CLI `--yes`) and a price cap checked against the live offer;
  without it, only the plan is printed.
- Every rental has a deadline, enforced twice: the `guard` (on the laptop) destroys the instance
  at the time or budget limit, on Ctrl-C and when the instance dies; and a backstop in the
  instance's start script destroys the instance from inside, 10 minutes after the deadline,
  with the per-instance key Vast.ai sets in every container (`CONTAINER_API_KEY`,
  `CONTAINER_ID`), in case the laptop sleeps or loses the network.
- Destroy, never stop: a stopped instance keeps billing its disk.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

API = "https://console.vast.ai"
DEAD_STATES = {"exited", "offline", "unknown"}  # Vast: an instance in these never reaches running


class VastError(RuntimeError):
    pass


# Start script of the instance. $CONTAINER_ID and $CONTAINER_API_KEY are set by Vast.ai.
BACKSTOP = """cat > "$HOME/backstop.sh" <<'END'
sleep {seconds}
export URL="${VAST_API_URL:-https://console.vast.ai}/api/v0/instances/$CONTAINER_ID/"
for i in 1 2 3 4 5; do
  curl -sf -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" "$URL" && exit 0
  python3 -c 'import os, urllib.request as u
u.urlopen(u.Request(os.environ["URL"], method="DELETE",
  headers={"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"]}), timeout=30)' && exit 0
  sleep 60
done
END
nohup bash "$HOME/backstop.sh" > "$HOME/backstop.log" 2>&1 &
"""


@dataclass(frozen=True)
class OfferQuery:
    """Search filters, protocol §2: verified host, reliability, disk and download floors."""

    gpu_name: str = "A40"
    num_gpus: int = 2
    min_reliability: float = 0.98
    disk_gb: int = 150  # weights 70.8 GB + image + logs; also the disk asked at creation
    min_inet_down_mbps: float = 500
    min_cuda: float = 12.8
    max_dph: float | None = None
    limit: int = 10

    def to_api(self) -> dict:
        q: dict = {
            "gpu_name": {"eq": self.gpu_name},
            "num_gpus": {"eq": self.num_gpus},
            "rentable": {"eq": True},
            "verified": {"eq": True},
            "reliability2": {"gte": self.min_reliability},
            "disk_space": {"gte": self.disk_gb},
            "inet_down": {"gte": self.min_inet_down_mbps},
            "cuda_max_good": {"gte": self.min_cuda},
            "allocated_storage": self.disk_gb,  # dph_total then includes this disk
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": self.limit,
        }
        if self.gpu_name.lower() == "any":  # e.g. the cheapest 1-GPU node for the backstop check
            del q["gpu_name"]
        if self.max_dph is not None:
            q["dph_total"] = {"lte": self.max_dph}
        return q


@dataclass(frozen=True)
class Offer:
    id: int
    gpu_name: str
    num_gpus: int
    dph_total: float  # USD per hour, GPUs and the asked disk
    gpu_ram_gb: float
    reliability: float
    inet_down_mbps: float
    disk_gb: float
    cuda: float
    pci_gen: float | None
    geolocation: str
    machine_id: int | None

    @classmethod
    def from_api(cls, d: dict) -> Offer:
        return cls(
            id=int(d["id"]),
            gpu_name=d.get("gpu_name", ""),
            num_gpus=int(d.get("num_gpus", 0)),
            dph_total=float(d["dph_total"]),
            gpu_ram_gb=round(float(d.get("gpu_ram", 0)) / 1024, 1),
            reliability=float(d.get("reliability2") or d.get("reliability") or 0),
            inet_down_mbps=float(d.get("inet_down") or 0),
            disk_gb=float(d.get("disk_space") or 0),
            cuda=float(d.get("cuda_max_good") or 0),
            pci_gen=d.get("pci_gen"),
            geolocation=d.get("geolocation") or "",
            machine_id=d.get("machine_id"),
        )


@dataclass(frozen=True)
class RentSpec:
    image: str = "vllm/vllm-openai:v0.32.0"  # the pinned vLLM version of the protocol
    disk_gb: int = 150
    label: str = "hpc-sizing"
    max_hours: float = 1.0
    env: dict[str, str] = field(default_factory=dict, repr=False)  # values never printed

    backstop_margin_s: int = 600

    def onstart(self) -> str:
        """Start script: the backstop destroys the instance from inside after the deadline.

        It uses the instance's own key (set by Vast.ai in every container), never the owner's,
        tries curl then Python, and retries for 5 minutes.
        """
        seconds = int(self.max_hours * 3600) + self.backstop_margin_s
        return BACKSTOP.replace("{seconds}", str(seconds))

    def body(self) -> dict:
        return {
            "client_id": "me",
            "image": self.image,
            "disk": self.disk_gb,
            "label": self.label,
            "runtype": "ssh",
            "onstart": self.onstart(),
            "env": dict(self.env),
        }

    def plan(self) -> dict:
        """The creation request with environment values masked, safe to print."""
        body = self.body()
        body["env"] = {k: "***" for k in self.env}
        return body


class VastClient:
    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str = API,
    ):
        self._key = api_key if api_key is not None else os.environ.get("VAST_API_KEY") or None
        self._http = httpx.Client(base_url=base_url, transport=transport, timeout=30)

    def __repr__(self) -> str:
        return f"VastClient(key={'set' if self._key else 'unset'})"

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    def _request(
        self, method: str, path: str, *, auth: bool = True, allow_empty: bool = False, **kwargs
    ) -> dict:
        headers = {}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        elif auth:
            raise VastError("VAST_API_KEY is not set: the owner sets it in the environment")
        try:
            resp = self._http.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as err:
            raise VastError(f"{method} {path}: {type(err).__name__}") from err
        # A redirect (the API moves paths without their trailing slash) or a non-JSON answer is
        # an error, never an empty result: an empty instance list would end the guard.
        if resp.status_code >= 300:
            raise VastError(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:200]}")
        if allow_empty and not resp.content.strip():  # e.g. 204 to a destroy
            return {}
        try:
            data = resp.json()
        except ValueError:
            raise VastError(f"{method} {path}: answer is not JSON") from None
        if not isinstance(data, dict):
            raise VastError(f"{method} {path}: unexpected answer {str(data)[:200]}")
        return data

    # Search works without a key (public offers); the key, when set, is sent anyway.
    def search(self, query: OfferQuery) -> list[Offer]:
        data = self._request(
            "GET", "/api/v0/bundles/", auth=False, params={"q": json.dumps(query.to_api())}
        )
        offers = [Offer.from_api(o) for o in data.get("offers", [])]
        if query.max_dph is not None:
            offers = [o for o in offers if o.dph_total <= query.max_dph]
        return sorted(offers, key=lambda o: o.dph_total)

    def offer(self, offer_id: int, disk_gb: int) -> Offer | None:
        """The live state of one offer (price with that disk), or None if it is gone."""
        # Searching by "id" returns nothing; the offer id is also its ask contract id.
        q = {"ask_contract_id": {"eq": offer_id}, "allocated_storage": disk_gb, "type": "on-demand"}
        data = self._request("GET", "/api/v0/bundles/", auth=False, params={"q": json.dumps(q)})
        offers = [
            Offer.from_api(o)
            for o in data.get("offers", [])
            if int(o["id"]) == offer_id and o.get("rentable", True)
        ]
        return offers[0] if offers else None

    def create(self, offer_id: int, spec: RentSpec) -> int:
        data = self._request("PUT", f"/api/v0/asks/{offer_id}/", json=spec.body())
        if not data.get("success") or "new_contract" not in data:
            raise VastError(f"creation refused: {str(data)[:200]}")
        return int(data["new_contract"])

    def show(self, instance_id: int) -> dict:
        return self._request("GET", f"/api/v0/instances/{instance_id}/").get("instances") or {}

    def instances(self) -> list[dict]:
        return self._request("GET", "/api/v1/instances/", params={"limit": 25}).get("instances", [])

    def destroy(self, instance_id: int) -> None:
        data = self._request("DELETE", f"/api/v0/instances/{instance_id}/", allow_empty=True)
        if data.get("success") is False:  # a refusal can come with HTTP 200
            raise VastError(f"destroy refused: {str(data)[:200]}")

    def balance(self) -> float:
        """Prepaid credit left (`credit`; the API's `balance` is the amount owed)."""
        data = self._request("GET", "/api/v0/users/current/")
        if data.get("credit") is None:
            raise VastError("account answer has no credit field")
        return float(data["credit"])


@dataclass(frozen=True)
class Rental:
    instance_id: int
    dph_total: float  # live price of the offer when it was rented


def rent(
    client: VastClient,
    offer_id: int,
    spec: RentSpec,
    max_dph: float,
    confirm: bool,
    log: Callable[[str], None] = print,
) -> Rental | None:
    """Rent one offer if its live price is within `max_dph`; without `confirm`, plan only."""
    offer = client.offer(offer_id, spec.disk_gb)
    if offer is None:
        raise VastError(f"offer {offer_id} is no longer available")
    if offer.dph_total > max_dph:
        raise VastError(f"offer {offer_id} costs {offer.dph_total:.3f} USD/h > cap {max_dph}")
    log(f"offer {offer_id}: {offer.num_gpus} x {offer.gpu_name}, {offer.dph_total:.3f} USD/h")
    log(f"request: {json.dumps(spec.plan())}")
    if not confirm:
        log("dry run: nothing rented (add --yes to rent)")
        return None
    try:
        instance_id = client.create(offer_id, spec)
    except VastError:
        # the answer may be lost after the instance was made: say where to look
        log("creation failed; an instance may exist anyway: check `hpc-sizing vast list` now")
        raise
    log(f"rented instance {instance_id}; the guard must run until it is destroyed")
    return Rental(instance_id, offer.dph_total)


@dataclass
class GuardResult:
    reason: str
    destroyed: bool
    elapsed_h: float
    spent_usd: float


def guard(
    client: VastClient,
    instance_id: int,
    max_hours: float,
    budget_usd: float,
    poll_s: float = 60,
    dead_grace_s: float = 600,
    price_hint: float = 0.0,
    count_instance_age: bool = True,
    reserve_usd: float = 0.0,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> GuardResult:
    """Watch the instance and destroy it at the deadline, the budget, on Ctrl-C or SIGTERM,
    or when it stays dead (exited, offline, unknown) for `dead_grace_s`.

    With `count_instance_age`, time counts from the instance's creation (`start_date`, checked
    on the test session), so a guard restarted on a running instance does not grant it a new
    deadline; `rent` turns it off for the guard of a fresh rental (age 0 by construction).
    Spending uses the instance's price, or `price_hint` until it is read; `reserve_usd` is kept
    out of the budget for what the guard cannot see, such as bandwidth. Any unexpected error
    destroys too.
    """
    start = clock()
    price = price_hint
    age_s = 0.0  # instance age at the last read, minus the guard time at that read
    status = None
    dead_since: float | None = None
    stop = {"signal": None}

    def on_signal(signum, _frame):
        stop["signal"] = signal.Signals(signum).name

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    reason = ""
    try:
        try:
            while True:
                elapsed = clock() - start + age_s
                if stop["signal"]:
                    reason = f"{stop['signal']} received"
                    break
                try:
                    info = client.show(instance_id)
                except VastError as err:
                    log(f"guard: cannot read the instance ({err}); retrying")
                    info = None
                if info is not None and not info and not _listed(client, instance_id):
                    # The API may not report a live instance yet (just after creation): send a
                    # destroy anyway; it fails only if the instance is really gone.
                    hours = elapsed / 3600
                    try:
                        client.destroy(instance_id)
                    except VastError:
                        return GuardResult("instance no longer exists", False, hours, hours * price)
                    reason = "instance not reported by the API; destroyed to be sure"
                    log(f"guard: {reason}")
                    return GuardResult(reason, True, hours, hours * price)
                if info:
                    price = float(info.get("dph_total") or price)
                    if count_instance_age and info.get("start_date"):
                        age_s = max(age_s, wall() - float(info["start_date"]) - (clock() - start))
                        elapsed = clock() - start + age_s
                    status = info.get("actual_status")
                    if status in DEAD_STATES:
                        dead_since = dead_since if dead_since is not None else clock()
                        if clock() - dead_since >= dead_grace_s:
                            reason = f"instance {status} for {dead_grace_s:.0f} s"
                            break
                    else:
                        dead_since = None
                spent = elapsed / 3600 * price
                if elapsed >= max_hours * 3600:
                    reason = f"time limit ({max_hours} h)"
                    break
                if spent + reserve_usd >= budget_usd:
                    reason = f"budget ({spent:.2f} + {reserve_usd:g} reserved >= {budget_usd} USD)"
                    break
                log(f"guard: {elapsed / 3600:.2f} h, {spent:.2f} USD, status {status}")
                for _ in range(max(1, int(poll_s))):  # 1 s slices: Ctrl-C acts within a second
                    if stop["signal"]:
                        break
                    sleep(min(1.0, poll_s))
        except Exception as err:  # a bug or odd API data must not leave the instance running
            reason = f"guard error: {type(err).__name__}: {err}"
            log(reason)
        elapsed = clock() - start + age_s
        destroyed = destroy_with_retries(client, instance_id, log=log, sleep=sleep)
        return GuardResult(reason, destroyed, elapsed / 3600, elapsed / 3600 * price)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def _listed(client: VastClient, instance_id: int) -> bool:
    """True unless the instance list, read successfully, does not contain it."""
    try:
        return any(int(i.get("id", -1)) == instance_id for i in client.instances())
    except VastError:
        return True  # cannot confirm it is gone: keep guarding


def destroy_with_retries(
    client: VastClient,
    instance_id: int,
    attempts: int = 5,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            client.destroy(instance_id)
            log(f"instance {instance_id} destroyed")
            return True
        except VastError as err:
            log(f"destroy attempt {attempt}/{attempts} failed: {err}")
            sleep(min(30, 2**attempt))
    log(
        f"DESTROY FAILED for instance {instance_id}: destroy it by hand at "
        "https://cloud.vast.ai/instances/ now"
    )
    return False


def ssh_command(info: dict) -> str | None:
    host, port = info.get("ssh_host"), info.get("ssh_port")
    if not host or not port:
        return None
    return shlex.join(["ssh", "-p", str(port), f"root@{host}"])


def offer_row(o: Offer) -> str:
    return (
        f"{o.id:>10}  {o.num_gpus} x {o.gpu_name:<6} {o.dph_total:6.3f} $/h  "
        f"rel {o.reliability:.3f}  down {o.inet_down_mbps:7.0f} Mb/s  disk {o.disk_gb:5.0f} GB  "
        f"CUDA {o.cuda}  PCIe {o.pci_gen}  {o.geolocation}"
    )
