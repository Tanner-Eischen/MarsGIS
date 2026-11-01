async(page)=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const terminal=page.waitForResponse(async r=>r.request().method()==='GET'&&/\/api\/v1\/explorer\/routes\/[a-zA-Z0-9_-]+$/.test(new URL(r.url()).pathname)&&(await r.json()).state!=='running',{timeout:45000});
 await page.getByRole('button',{name:'Plan route',exact:true}).click();const job=await(await terminal).json();if(job.state!=='ready')throw Error(job.error);
 await page.getByRole('heading',{name:'Route results',exact:true}).waitFor();
 let fallback=false;try{await page.getByRole('alert').waitFor({timeout:5000});fallback=true;}catch{}
 const result={case:'B09',status:fallback&&errors.length===0?'passed':'failed',unavailable_webgl_control:true,map_available:await page.getByRole('region',{name:'Interactive Mars terrain map',exact:true}).isVisible(),download_available:await page.getByRole('link',{name:'Download this route',exact:true}).isVisible(),fallback_alert:fallback,alert_text:fallback?await page.getByRole('alert').innerText():null,unhandled_errors:errors};
 await page.screenshot({path:'output/playwright/wave2-20261101T101242Z/final/no-webgl.png',fullPage:true});return result;
}

