async (page) => {
 const identity=await (await page.request.get('http://127.0.0.1:18534/api/v1/health/live')).json();
 if(identity.build?.backend_sha256!=='197047df690db53a94dbe14776919718d03708e362ec25ef41c87d22942ff5e4')throw Error('Wrong service');
 await page.getByRole('button',{name:'Clear points',exact:true}).click();
 const map=page.locator('.leaflet-container');const box=await map.boundingBox();
 await page.mouse.click(box.x+box.width/2-12,box.y+box.height/2);
 const a={lat:await page.getByLabel('Start latitude',{exact:true}).inputValue(),lon:await page.getByLabel('Start longitude',{exact:true}).inputValue()};
 await page.mouse.click(box.x+box.width/2+12,box.y+box.height/2+5);
 const b={lat:await page.getByLabel('Destination latitude',{exact:true}).inputValue(),lon:await page.getByLabel('Destination longitude',{exact:true}).inputValue()};
 if(!a.lat||!b.lat||a.lon===b.lon)throw Error('Map clicks did not set distinct A/B');
 return {case:'B02',status:'passed',a,b,box};
}

