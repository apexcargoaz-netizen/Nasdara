// Tenderloin Nasdara Map: keeps a copy so the map opens with bad signal. Always tries the network first, so updates come through.
const C="nasdara-v2",CT="nasdara-tiles-v1";
self.addEventListener("install",e=>{self.skipWaiting();e.waitUntil(caches.open(C).then(c=>c.addAll(["./","./manifest.webmanifest","./icon.png"]).catch(()=>{})))});
self.addEventListener("activate",e=>e.waitUntil((async()=>{
  const keys=await caches.keys();await Promise.all(keys.filter(k=>k!==C&&k!==CT).map(k=>caches.delete(k)));  // wipe old app caches so a stale page can't stick
  await self.clients.claim();
  const cs=await self.clients.matchAll({type:"window"});cs.forEach(c=>c.postMessage({t:"sw-updated"}));  // tell open tabs to reload onto the fresh copy
})()));
self.addEventListener("fetch",e=>{const r=e.request;if(r.method!=="GET")return;const u=new URL(r.url);
  const mine=u.origin===location.origin,cdn=/cdn\.jsdelivr\.net$/.test(u.hostname);if(!mine&&!cdn)return;
  if(mine&&u.searchParams.has("vc"))return;
  if(mine&&u.pathname.includes("/tiles/")){e.respondWith(caches.open(CT).then(c=>c.match(r).then(m=>m||fetch(r).then(res=>{if(res.ok)c.put(r,res.clone());return res}))));return}   // map tiles: saved once, reused (the ?t= stamp changes when tiles are redone)
  const nav=r.mode==="navigate";   // the app shell + scripts: always hit the network fresh, bypassing the browser's HTTP cache, so a new version always comes through
  e.respondWith(fetch(nav?new Request(r.url,{cache:"reload"}):r).then(res=>{if(res.ok&&(mine||cdn)){const cp=res.clone();caches.open(C).then(c=>c.put(mine&&nav?"./":r,cp))}return res})
    .catch(()=>caches.match(mine&&nav?"./":r).then(m=>m||(nav?caches.match("./"):Response.error()))))});   // only fall back to the app page for a page load, never for a failed script/asset
