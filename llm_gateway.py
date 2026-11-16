import asyncio, random, time
from dataclasses import dataclass


# ---------- Provided: do not modify ----------

class TransientError(Exception): ...
class RateLimitError(TransientError): ...
class ProviderUnavailable(TransientError): ...
class InvalidRequestError(Exception): ...


class MockProvider:
    def __init__(self, name, fail_rate=0.3, latency=(0.1, 1.5), max_concurrency=5):
        self.name, self.fail_rate, self.latency = name, fail_rate, latency
        self.max_concurrency = max_concurrency
        self._in_flight = 0
        self.peak_in_flight = 0

    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise InvalidRequestError("empty prompt")
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self._in_flight > self.max_concurrency:
                raise RateLimitError(f"{self.name}: concurrency exceeded")
            await asyncio.sleep(random.uniform(*self.latency))
            r = random.random()
            if r < self.fail_rate / 2:
                raise RateLimitError(f"{self.name}: 429")
            if r < self.fail_rate:
                raise ProviderUnavailable(f"{self.name}: 503")
            return f"[{self.name}] answer to: {prompt[:30]}"
        finally:
            self._in_flight -= 1


@dataclass
class Result:
    text: str
    provider: str
    attempts: int
    latency_ms: float


class AllProvidersFailed(Exception):
    def __init__(self, errors):
        super().__init__(f"all providers failed: {errors}")
        self.errors = errors


# ---------- Your code ----------

class _CircuitBreaker:
    """Bonus: skip a provider for `cooldown_s` after `threshold` consecutive failures.

    States:
      closed    - calls pass; failures are counted.
      open      - calls are skipped until `open_until`.
      half-open - after the cooldown, calls pass again. One more failure re-opens
                  the circuit at once (the counter is still >= threshold).
                  One success closes it (the counter resets to 0).
    """

    def __init__(self, threshold=5, cooldown_s=10.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.consecutive_failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        return time.monotonic() >= self.open_until

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown_s


class LLMGateway:
    MAX_BACKOFF_S = 2.0

    def __init__(self, providers, timeout_s=1.0, max_retries=2, base_backoff_s=0.1):
        self.providers = list(providers)  # priority order: index 0 first
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        # One semaphore per provider enforces the concurrency cap (requirement 5).
        self._semaphores = {p.name: asyncio.Semaphore(p.max_concurrency) for p in self.providers}
        self._breakers = {p.name: _CircuitBreaker() for p in self.providers}

    def _backoff_delay(self, retry_index: int) -> float:
        # Exponential backoff with "full jitter": uniform(0, base * 2^n), capped.
        ceiling = min(self.MAX_BACKOFF_S, self.base_backoff_s * (2 ** retry_index))
        return random.uniform(0, ceiling)

    async def _attempt(self, provider, prompt: str) -> str:
        # Take the semaphore first, so time spent waiting in the queue does not
        # count against the per-attempt timeout. The timeout covers only the call.
        async with self._semaphores[provider.name]:
            # wait_for cancels the coroutine on timeout. Its `finally` block runs,
            # so the provider's in-flight count drops and the slot is released.
            return await asyncio.wait_for(provider.generate(prompt), timeout=self.timeout_s)

    async def complete(self, prompt: str) -> Result:
        start = time.perf_counter()
        attempts = 0
        errors = []  # history: (provider, attempt_no, error description)

        for provider in self.providers:  # requirement 1: fallback in priority order
            breaker = self._breakers[provider.name]

            for retry_index in range(self.max_retries + 1):  # 1 try + max_retries retries
                if not breaker.allow():
                    errors.append((provider.name, None, "circuit open: skipped"))
                    break  # skip to the next provider

                attempts += 1
                try:
                    text = await self._attempt(provider, prompt)
                except InvalidRequestError:
                    raise  # requirement 4: fail fast, no retry, no fallback
                except (TransientError, asyncio.TimeoutError) as exc:  # requirement 3
                    breaker.record_failure()
                    desc = "timeout" if isinstance(exc, asyncio.TimeoutError) else repr(exc)
                    errors.append((provider.name, attempts, desc))
                    if retry_index < self.max_retries:  # no sleep after the last try
                        await asyncio.sleep(self._backoff_delay(retry_index))
                    continue
                else:
                    breaker.record_success()
                    latency_ms = (time.perf_counter() - start) * 1000
                    return Result(text, provider.name, attempts, latency_ms)  # requirement 6

        raise AllProvidersFailed(errors)  # requirement 6


# ---------- Harness: do not modify ----------

async def main():
    primary = MockProvider("primary", fail_rate=0.4, latency=(0.1, 1.5), max_concurrency=5)
    fallback = MockProvider("fallback", fail_rate=0.1, latency=(0.2, 0.6), max_concurrency=10)
    gw = LLMGateway([primary, fallback])
    prompts = [f"question {i}" for i in range(50)] + [""]
    results = await asyncio.gather(*(gw.complete(p) for p in prompts), return_exceptions=True)
    ok = [r for r in results if isinstance(r, Result)]
    print(f"success: {len(ok)}/{len(prompts)}")
    print("by provider:", {p: sum(r.provider == p for r in ok) for p in ("primary", "fallback")})
    print("errors:", [type(r).__name__ for r in results if isinstance(r, Exception)])
    print("peak in-flight:", primary.peak_in_flight, fallback.peak_in_flight)
    assert primary.peak_in_flight <= 5 and fallback.peak_in_flight <= 10, "concurrency cap violated"


asyncio.run(main())

# Before production:
# 1. Add an end-to-end deadline (a total budget across all providers, including
#    semaphore wait time), plus metrics/tracing per attempt (provider, error, latency).
# 2. Honour Retry-After headers on 429s, bound the semaphore queue (shed load when
#    it is full), and make the breaker safe across processes (e.g. shared state).
# 3. Make timeouts, retries and breaker settings configurable per provider, and add
#    tests with deterministic fakes (fixed seeds, fake clock) instead of random mocks.
