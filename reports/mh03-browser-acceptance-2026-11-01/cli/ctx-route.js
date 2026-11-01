async (page) => {
 const identity=await (await page.request.get('http://127.0.0.1:18534/api/v1/health/live')).json();
 if(identity.build?.backend_sha256!=='197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4')throw Error('Wrong service');
 page.__marsEvidence ||= {errors:[],posts:[],jobs:[]};
 if(!page.__marsListeners){page.__marsListeners=true;page.on('pageerror',e=>page.__marsEvidence.errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/v1/explorer/routes'))page.__marsEvidence.posts.push(r.postDataJSON());});}
 const terminal=page.waitForResponse(async r=>r.request().method()==='GET'&&/\/api\/v1\/explorer\/routes\/[a-zA-Z0-9_-]+$/.test(new URL(r.url()).pathname)&&(await r.json()).state!=='running',{timeout:45000});
 await page.getByRole('button',{name:'Plan route',exact:true}).click();
 const job=await (await terminal).json();page.__marsEvidence.jobs.push(job);page.__lastMarsJob=job;
 if(job.state!=='ready')throw Error(job.error);
 if(job.id!==job.result.id||job.request_id!==job.result.request_id)throw Error('Request/result mismatch');
 await page.getByRole('heading',{name:'Route results',exact:true}).waitFor();
 await page.locator('.js-plotly-plot').waitFor({state:'visible',timeout:20000});
 const scene=await page.evaluate(()=>{const g=document.querySelector('.js-plotly-plot'); const s=g?._fullLayout?.scene?._scene;return {canvasCount:g?.querySelectorAll('canvas').length,sceneKeys:s?Object.keys(s):[],glplotKeys:s?.glplot?Object.keys(s.glplot):[],camera:g?._fullLayout?.scene?.camera,aspect:g?._fullLayout?.scene?.aspectratio,dataTraceTypes:g?.data?.map(t=>t.type)};});
 return {case:'B03-B06',status:'passed',id:job.id,request_id:job.request_id,source:job.result.source.id,spacing_m:job.result.source.analysis_resolution_m,distance_m:job.result.distance_m,scene,errors:page.__marsEvidence.errors};
}

