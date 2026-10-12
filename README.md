# The Statistics Library

Eight interactive courses plus a Python capstone, packaged as one offline-first app.
No build step, no dependencies, no network at runtime. Total size: **~800 KB**.

```
stats-library/
├── index.html                  the hub: launcher, progress, review schedule
├── manifest.webmanifest        makes it installable
├── sw.js                       service worker — precaches everything for offline
├── shared/course-shell.js      injected into each course: progress capture + Library button
├── icons/                      app icons (192, 512, maskable, apple-touch)
├── courses/                    the eight courses
│   ├── level-zero.html
│   ├── bayesian-mastery.html
│   ├── bayesian-gym.html
│   ├── statistics-for-ml.html
│   ├── ml-gym.html
│   ├── multivariate-linear-algebra.html
│   ├── latent-variables-vi.html
│   └── statistics-of-deep-learning.html
├── capstone/                   the Python project (runs on a computer, not a phone)
└── native/                     optional: Capacitor wrapper for the App Store / Play Store
```

---

## 1. Run it (mobile web app)

The service worker and installability both require `http://` or `https://` — opening
`index.html` straight from the filesystem works for reading, but not for offline
install or progress saving.

**Quickest local test:**

```bash
cd stats-library
python3 -m http.server 8080
```

Then open `http://localhost:8080` — or `http://<your-mac's-LAN-ip>:8080` from your
phone on the same Wi-Fi.

**Permanent hosting on your Mac Mini via Cloudflare Tunnel** — the same pattern your
other products use. Serve the folder and point a hostname at it:

```bash
cd /path/to/stats-library
python3 -m http.server 8080
cloudflared tunnel --url http://localhost:8080
```

For something you'll keep, add it to your existing tunnel config as its own hostname
(for example `learn.fueldeskpro.com`) rather than using a quick tunnel. Any static
host works too — Cloudflare Pages, GitHub Pages, S3 — since there is no server-side
component at all.

A real HTTPS hostname matters for one reason: **service workers only run on HTTPS or
localhost.** Without it you lose offline caching (the courses still work online).

---

## 2. Install it on your phone

**iPhone / iPad** — open the URL in **Safari** (not Chrome; only Safari can install on
iOS), tap Share, then **Add to Home Screen**. It launches full-screen with no browser
chrome, and after the first visit every course is cached on the device.

**Android** — open in Chrome and tap **Install app** from the menu, or accept the
install prompt. Same result.

Verified behaviour once installed: the hub and all eight courses load with the network
completely off.

---

## 3. Progress, and its honest limits

Progress is captured by watching each course's own level-completion state, so the
courses were not rewritten — the shared script only observes.

- Stored in `localStorage` **on that device, in that browser.** It does not sync
  between your phone and your laptop, and it is not sent anywhere.
- **Export / Import** buttons on the hub move a small JSON file between devices. Use
  them as your backup: browsers can and do evict site storage under pressure, and iOS
  is more aggressive about this than desktop. If a hundred hours of progress would
  annoy you to lose, export it occasionally.
- Progress is monotonic — it unions with what was recorded before, so it never goes
  backwards if a course is reloaded mid-session.

Note that the courses themselves still keep *in-page* state per session (which level
you're on, which quizzes you answered). The library tracks completion, not scroll
position.

---

## 4. Updating a course

Edit the file in `courses/`, then **bump the version string in `sw.js`**:

```js
var VERSION = 'stats-library-v2';   // was v1
```

Without that bump, devices with the old version cached will keep serving it — the
cache is deliberately cache-first, which is what makes it fast and offline-capable.
Changing `VERSION` deletes the old cache and re-precaches on next load.

If you regenerate a course from the original single-file versions, re-apply the three
packaging patches: the `data-course="<slug>"` attribute on `<html>`, the mobile/PWA
meta block, and the `shared/course-shell.js` script tag before `</body>`.

---

## 5. Native iOS / Android apps (optional)

**Read this first: you probably don't need it.** An installed PWA gives you a home
screen icon, full-screen launch, and complete offline use — everything a native
wrapper would, minus the store listing. Store submission is worth it only if you want
to *distribute* this to other people.

If you do, `native/` contains a [Capacitor](https://capacitorjs.com) scaffold that
wraps these exact files.

### What you need
- **Android:** Android Studio, JDK 17. Free to build; **$25 one-time** for a Play
  Console account to publish.
- **iOS:** a Mac (you have one), Xcode, and an **Apple Developer account at $99/year**
  to ship to the App Store. Without it you can still build to your own device with a
  free account, but the build expires after 7 days.

### Steps

```bash
cd stats-library/native
npm install
bash sync-web.sh          # copies the web app into native/www
npx cap add android
npx cap add ios
npx cap sync
npx cap open android      # builds/runs from Android Studio
npx cap open ios          # builds/runs from Xcode
```

After any change to the web files, re-run `bash sync-web.sh && npx cap sync`.

Set your own `appId` in `capacitor.config.json` before publishing — the placeholder is
`com.fueldeskpro.statslibrary`.

### App icons and splash screens
Generate the full native icon sets from the 512px icon:

```bash
npm install -D @capacitor/assets
mkdir -p assets && cp ../icons/icon-512.png assets/icon.png
npx capacitor-assets generate
```

### A candid word about store review
Apple's guideline 4.2 ("Minimum Functionality") rejects apps that are essentially a
website in a wrapper. This bundle has a reasonable case — it is fully offline,
interactive, and stores data locally, which is materially more than a web view of a
site. But it is a real risk, and reviewers vary. If a rejection would be a waste of
your time, the PWA route avoids the question entirely.

**Simpler alternative for Android:** [PWABuilder](https://www.pwabuilder.com) will
package this PWA as a Trusted Web Activity for Play with no native code at all. Google
is explicitly friendly to that approach; Apple is not.

---

## 6. The Python capstone

`capstone/` (bundled here) is the one piece that
doesn't run in a browser. It needs a real Python environment:

```bash
cd capstone
pip install numpy scipy matplotlib scikit-learn pandas
python3 station_capstone.py              # runs on included synthetic data
python3 station_capstone.py my_data.csv  # your data: date,station_id,visitors,upsells,promo
```

It writes `report.md` and six charts. This is the exercise that makes everything else
permanent.
