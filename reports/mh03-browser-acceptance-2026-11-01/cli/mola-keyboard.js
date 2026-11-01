async(page)=>{
 const identity=await(await page.request.get('http://127.0.0.1:18534/api/v1/health/live')).json();if(identity.build?.backend_sha256!=='197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4')throw Error('Wrong service');
 const fields=[['Start latitude','10'],['Start longitude','179'],['Destination latitude','12'],['Destination longitude','-179']];
 await page.getByLabel(fields[0][0],{exact:true}).focus();const focus=[];
 for(let i=0;i<fields.length;i++){const label=await page.evaluate(()=>document.activeElement.getAttribute('aria-label'));focus.push(label);if(label!==fields[i][0])throw Error('Keyboard tab order mismatch');await page.keyboard.press('ControlOrMeta+A');await page.keyboard.type(fields[i][1]);await page.keyboard.press('Tab');}
 await page.keyboard.press('Tab');if((await page.evaluate(()=>document.activeElement.textContent)).trim()!=='Plan route')throw Error('Submit is not keyboard reachable');
 const terminal=page.waitForResponse(async r=>r.request().method()==='GET'&&/\/api\/v1\/explorer\/routes\/[a-zA-Z0-9_-]+$/.test(new URL(r.url()).pathname)&&(await r.json()).state!=='running',{timeout:45000});
 await page.keyboard.press('Enter');const job=await(await terminal).json();if(job.state!=='ready'||job.result.source.id!=='mola16')throw Error(job.error||'Wrong source');page.__marsEvidence.jobs.push(job);page.__lastMarsJob=job;
 await page.getByRole('heading',{name:'Route results',exact:true}).waitFor();
 return {case:'B04-B10',status:'passed',keyboard_focus_order:focus,submitted_by_enter:true,id:job.id,request_id:job.request_id,source:job.result.source.id,label:await page.locator('.mars-source-badge').innerText(),spacing_m:job.result.source.analysis_resolution_m,distance_m:job.result.distance_m,warning:job.result.warning};
}

