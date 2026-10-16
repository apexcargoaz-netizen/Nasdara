// Tenderloin Nasdara Map: keeps a copy so the map opens with bad signal. Always tries the network first, so updates come through.
const C="nasdara-v1";
self.addEventListener("install",e=>{self.skipWaiting();e.waitUntil(caches.open(C).then(c=>c.addAll(["./","./manifest.webmanifest","./icon.png"]).catch(()=>{})))});
self.addEventListener("activate",e=>e.waitUntil(self.clients.claim()));
self.addEventListener("fetch",e=>{const r=e.request;if(r.method!=="GET")return;const u=new URL(r.url);
  const mine=u.origin===location.origin,cdn=/cdn\.jsdelivr\.net$/.test(u.hostname);if(!mine&&!cdn)return;
  if(mine&&u.searchParams.has("vc"))return;   // the "new version?" check must always hit the network
  e.respondWith(fetch(r).then(res=>{if(res.ok&&(mine||cdn)){const cp=res.clone();caches.open(C).then(c=>c.put(mine&&r.mode==="navigate"?"./":r,cp))}return res})
    .catch(()=>caches.match(mine&&r.mode==="navigate"?"./":r).then(m=>m||caches.match("./"))))});
