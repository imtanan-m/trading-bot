// Network-first: always try fresh data/app, fall back to the last copy when offline.
const C = 'paperdesk-v1';
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET' || !e.request.url.startsWith(self.location.origin)) return;
  e.respondWith(fetch(e.request).then(r => {
    const copy = r.clone(); caches.open(C).then(c => c.put(e.request.url.split('?')[0], copy)); return r;
  }).catch(() => caches.match(e.request.url.split('?')[0])));
});
