async (page) => {
  const base = 'http://127.0.0.1:18534';
  const identity = await (await page.request.get(base + '/api/v1/health/live')).json();
  if (identity.build?.backend_sha256 !== '197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4') throw Error('Wrong service');
  const results = [];
  const waitDone = async (job) => {
    for (let n = 0; n < 100; n++) {
      const value = await (await page.request.get(base + job.poll_url)).json();
      if (value.state !== 'running') { if (value.state !== 'ready') throw Error(value.error); return value; }
      await page.waitForTimeout(50);
    }
    throw Error('Real server worker did not drain');
  };
  const ignoreTransportAbort = async () => page.evaluate(() => {
    window.__marsRealFetch = window.fetch;
    window.fetch = (input, options = {}) => {
      if (new URL(typeof input === 'string' ? input : input.url, location.href).pathname.startsWith('/api/v1/explorer/routes')) {
        const { signal, ...rest } = options;
        return window.__marsRealFetch(input, rest);
      }
      return window.__marsRealFetch(input, options);
    };
  });
  await ignoreTransportAbort();
  try {
    let release, arrived, delivered;
    const gate = new Promise(r => { release = r; });
    const firstResponse = new Promise(r => { arrived = r; });
    const delivery = new Promise(r => { delivered = r; });
    let first = true;
    const handler = async route => {
      if (route.request().method() !== 'POST' || !first) return route.continue();
      first = false;
      const real = await route.fetch();
      arrived(await real.json());
      await gate;
      await route.fulfill({ response: real });
      delivered();
    };
    await page.route('**/api/v1/explorer/routes', handler);
    await page.getByRole('button', { name: 'Use Jezero example', exact: true }).click();
    await page.getByRole('button', { name: 'Plan route', exact: true }).click();
    const old = await firstResponse;
    await waitDone(old);
    const input = page.getByLabel('Destination latitude', { exact: true });
    await input.fill('18.385027');
    const newest = page.waitForResponse(async r => r.request().method() === 'GET' && /\/api\/v1\/explorer\/routes\/[a-z0-9]+$/.test(new URL(r.url()).pathname) && (await r.json()).state === 'ready');
    await page.getByRole('button', { name: 'Plan route', exact: true }).click();
    const current = await (await newest).json();
    await page.getByRole('heading', { name: 'Route results', exact: true }).waitFor();
    release(); await delivery;
    await page.waitForTimeout(1800); // Covers a full 1200 ms polling interval after late delivery.
    const href = await page.getByRole('link', { name: 'Download this route', exact: true }).getAttribute('href');
    if (!href.includes(current.id) || href.includes(old.id)) throw Error('Late POST replaced the newer route');
    results.push({ case: 'out-of-order-real-POST', old_job: old.id, current_job: current.id, status: 'passed', response_values_unmodified: true, transport_abort_deliberately_ignored: true });
    await page.unroute('**/api/v1/explorer/routes', handler);

    let releasePoll, pollArrived, pollDelivered;
    const pollGate = new Promise(r => { releasePoll = r; });
    const terminalHeld = new Promise(r => { pollArrived = r; });
    const pollDelivery = new Promise(r => { pollDelivered = r; });
    let held = false;
    const pollHandler = async route => {
      if (route.request().method() !== 'GET') return route.continue();
      const response = await route.fetch();
      const job = await response.json();
      if (job.state !== 'ready' || held) return route.fulfill({ response });
      held = true; pollArrived(job);
      await pollGate; await route.fulfill({ response }); pollDelivered();
    };
    await page.route('**/api/v1/explorer/routes/*', pollHandler);
    await page.getByRole('button', { name: 'Use Jezero example', exact: true }).click();
    await page.getByRole('button', { name: 'Plan route', exact: true }).click();
    const abandoned = await terminalHeld;
    const original = await input.inputValue();
    await input.fill('18.385027'); await input.fill(original);
    releasePoll(); await pollDelivery;
    await page.waitForTimeout(1300);
    if (await page.getByRole('heading', { name: 'Route results', exact: true }).count()) throw Error('ABA revived an abandoned result');
    results.push({ case: 'ABA-real-terminal-poll', abandoned_job: abandoned.id, status: 'passed', response_values_unmodified: true, transport_abort_deliberately_ignored: true });
    await page.unroute('**/api/v1/explorer/routes/*', pollHandler);
  } finally {
    await page.unrouteAll({ behavior: 'ignoreErrors' });
    await page.evaluate(() => { if (window.__marsRealFetch) window.fetch = window.__marsRealFetch; delete window.__marsRealFetch; });
  }
  return { case: 'B08', status: 'passed', controls: results, observation: 'Real backend payloads held/reordered; only transport timing and abort delivery changed' };
}

