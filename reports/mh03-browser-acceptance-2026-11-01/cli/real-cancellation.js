async (page) => {
  const base = 'http://127.0.0.1:18534';
  const identity = await (await page.request.get(base + '/api/v1/health/live')).json();
  if (identity.build?.backend_sha256 !== '197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4') throw Error('Wrong service');
  let release, arrived;
  const gate = new Promise(r => { release = r; });
  const held = new Promise(r => { arrived = r; });
  const handler = async route => {
    const response = await route.fetch();
    arrived(await response.json());
    await gate;
    try { await route.fulfill({ response }); } catch { /* Browser already aborted. */ }
  };
  await page.route('**/api/v1/explorer/routes', handler);
  try {
    await page.getByRole('button', { name: 'Use Jezero example', exact: true }).click();
    const failed = page.waitForEvent('requestfailed', { predicate: r => r.method() === 'POST' && r.url().endsWith('/api/v1/explorer/routes') });
    await page.getByRole('button', { name: 'Plan route', exact: true }).click();
    const job = await held;
    await page.getByLabel('Destination latitude', { exact: true }).fill('18.385027');
    const request = await failed;
    release();
    if (await page.getByRole('heading', { name: 'Route results', exact: true }).count()) throw Error('Cancelled result visible');
    let terminal;
    for (let n = 0; n < 80; n++) {
      terminal = await (await page.request.get(base + job.poll_url)).json();
      if (terminal.state !== 'running') break;
      await page.waitForTimeout(50);
    }
    if (terminal.state !== 'ready') throw Error('Accepted worker did not drain safely');
    return { case: 'B08-real-cancellation', status: 'passed', browser_network_failure: request.failure()?.errorText, abandoned_job: job.id, server_worker_drained_to_ready: true, stale_result_visible: false };
  } finally {
    release();
    await page.unrouteAll({ behavior: 'ignoreErrors' });
  }
}
