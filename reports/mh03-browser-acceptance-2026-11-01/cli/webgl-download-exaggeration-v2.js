async (page) => {
 const identity=await (await page.request.get('http://127.0.0.1:18534/api/v1/health/live')).json();if(identity.build?.backend_sha256!=='197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4')throw Error('Wrong service');
 const job=page.__lastMarsJob;if(job?.state!=='ready')throw Error('No real computed route');
 const url='http://127.0.0.1:18534/api/v1/explorer/routes/'+job.id+'/geojson';
 const before=await (await page.request.get(url)).body();const posts=page.__marsEvidence.posts.length;
 const scientificBefore=await page.evaluate(()=>JSON.stringify(document.querySelector('.js-plotly-plot').data.map(t=>({x:t.x,y:t.y,z:t.z,customdata:t.customdata}))));
 const renderer=await page.evaluate(()=>{const gl=document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.glplot.gl;const ext=gl.getExtension('WEBGL_debug_renderer_info');return {lost:gl.isContextLost(),version:gl.getParameter(gl.VERSION),renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER)};});if(renderer.lost)throw Error('WebGL context lost');
 const downloadPromise=page.waitForEvent('download');await page.getByRole('link',{name:'Download this route',exact:true}).click();const download=await downloadPromise;
 await download.saveAs('output/playwright/wave2-20261101T101242Z/final/ctx-browser-download.geojson');if(await download.failure())throw Error('Browser download failed');
 const camera=()=>page.evaluate(()=>document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.getCamera());
 const cameraBefore=await camera();const box=await page.locator('.js-plotly-plot canvas').boundingBox();
 await page.mouse.move(box.x+box.width*.48,box.y+box.height*.55);await page.mouse.down();await page.mouse.move(box.x+box.width*.48+85,box.y+box.height*.55+30,{steps:12});await page.mouse.up();
 await page.waitForFunction(before=>JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.getCamera())!==before,JSON.stringify(cameraBefore));const cameraOrbit=await camera();
 await page.mouse.wheel(0,-300);await page.waitForFunction(before=>JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.getCamera())!==before,JSON.stringify(cameraOrbit));const cameraZoom=await camera();
 await page.getByRole('button',{name:'Reset 3D camera',exact:true}).click();
 await page.waitForFunction(()=>Math.abs(document.querySelector('.js-plotly-plot')._fullLayout.scene._scene.getCamera().eye.x-1.45)<1e-6);
 const slider=page.getByRole('slider',{name:'3D vertical exaggeration',exact:true}); const oldExag=Number(await slider.inputValue()); const oldAspect=await page.evaluate(()=>document.querySelector('.js-plotly-plot')._fullLayout.scene.aspectratio.z); await slider.focus();await slider.press('Home');
 await page.waitForFunction(()=>document.querySelector('input[aria-label="3D vertical exaggeration"]').value==='1');
 await page.waitForFunction(z=>Math.abs(document.querySelector('.js-plotly-plot')._fullLayout.scene.aspectratio.z-z)<1e-7,oldAspect/oldExag); const aspect1=await page.evaluate(()=>document.querySelector('.js-plotly-plot')._fullLayout.scene.aspectratio.z);
 await slider.press('End');await page.waitForFunction(()=>document.querySelector('input[aria-label="3D vertical exaggeration"]').value==='100');
 await page.waitForFunction(z=>Math.abs(document.querySelector('.js-plotly-plot')._fullLayout.scene.aspectratio.z/z-100)<1e-5,aspect1);
 const aspect100=await page.evaluate(()=>document.querySelector('.js-plotly-plot')._fullLayout.scene.aspectratio.z);
 const scientificAfter=await page.evaluate(()=>JSON.stringify(document.querySelector('.js-plotly-plot').data.map(t=>({x:t.x,y:t.y,z:t.z,customdata:t.customdata}))));
 const after=await (await page.request.get(url)).body();
 if(!before.equals(after)||scientificBefore!==scientificAfter||page.__marsEvidence.posts.length!==posts)throw Error('Exaggeration mutated science');
 return {case:'B05-B07',status:'passed',job_id:job.id,download_name:download.suggestedFilename(),download_bytes:before.length,renderer,cameraBefore,cameraOrbit,cameraZoom,aspect1,aspect100,export_unchanged:true,scientific_arrays_unchanged:true,extra_scientific_posts:0,unhandled_errors:page.__marsEvidence.errors};
}


