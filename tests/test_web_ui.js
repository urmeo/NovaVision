// Run with: node --test tests/test_web_ui.js (Node's built-in test runner).
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const html = readFileSync(join(__dirname, '..', 'static', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function createApp(fetch) {
    const elements = new Map();
    const downloads = [];
    const reportText = [];
    const reportImages = [];
    const clipboard = [];
    const errors = [];
    const timers = new Map();
    let nextTimer = 0;

    class Element {
        constructor(tag = 'div') {
            this.tag = tag;
            this.children = [];
            this.style = {};
            this.value = '';
            this.listeners = new Map();
            this.classes = new Set();
            this.classList = {
                add: (...names) => names.forEach(name => this.classes.add(name)),
                remove: (...names) => names.forEach(name => this.classes.delete(name)),
                contains: name => this.classes.has(name),
                toggle: (name, force = !this.classes.has(name)) => {
                    if (force) this.classes.add(name);
                    else this.classes.delete(name);
                    return force;
                }
            };
        }
        set className(value) { this.classes = new Set(value.split(/\s+/)); }
        get className() { return [...this.classes].join(' '); }
        set textContent(value) { this.text = String(value); }
        get textContent() { return this.text || ''; }
        set innerHTML(value) { this.html = value; this.children = []; }
        get innerHTML() { return this.html || ''; }
        append(...children) { this.children.push(...children); }
        appendChild(child) { this.append(child); }
        querySelector(selector) {
            this.parts ||= new Map();
            if (!this.parts.has(selector)) this.parts.set(selector, new Element());
            return this.parts.get(selector);
        }
        addEventListener(name, listener) {
            if (!this.listeners.has(name)) this.listeners.set(name, []);
            this.listeners.get(name).push(listener);
        }
        dispatch(name) { return this.listeners.get(name)?.forEach(listener => listener({})); }
        click() {
            if (this.tag === 'a') downloads.push({ href: this.href, filename: this.download });
            this.dispatch('click');
        }
        focus() {}
        getContext() {
            return new Proxy({}, {
                get: (_, name) => {
                    if (name === 'createLinearGradient') return () => ({ addColorStop() {} });
                    if (name === 'measureText') return text => ({ width: text.length * 7 });
                    if (name === 'fillText') return text => reportText.push(text);
                    if (name === 'drawImage') return image => reportImages.push(image.src);
                    return () => {};
                }
            });
        }
        toDataURL() { return 'data:image/png;report'; }
    }

    const getElementById = id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    };
    getElementById('liveEmotionBars').children = Array.from({ length: 7 }, () => new Element());
    getElementById('styleSelect').value = 'artistic';
    getElementById('generationBackend').textContent = html.match(/id="generationBackend"[^>]*>([^<]*)</)[1];
    const context = vm.createContext({
        document: {
            getElementById,
            createElement: tag => new Element(tag),
            querySelector: () => null,
            querySelectorAll: selector => selector === '.history-thumbnail'
                ? getElementById('historyThumbnails').children.filter(el => el.classList.contains('history-thumbnail'))
                : []
        },
        window: { scrollY: 0, addEventListener() {} },
        IntersectionObserver: class { observe() {} },
        Image: class {
            set src(value) { this.source = value; this.onload?.(); }
            get src() { return this.source; }
        },
        navigator: { clipboard: { writeText: async text => clipboard.push(text) } },
        performance: { now: () => 1000 },
        setTimeout: callback => { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout: id => timers.delete(id),
        setInterval: () => ++nextTimer,
        clearInterval() {},
        AbortController,
        fetch,
        console: { error: (...args) => errors.push(args) }
    });
    // Execute the whole shipped script, including the real input and history handlers.
    vm.runInContext(script, context, { filename: 'static/index.html' });
    const api = vm.runInContext('({ generateImage, analyzeLive, downloadImage, downloadReport })', context);
    return { api, getElementById, downloads, reportText, reportImages, clipboard, errors, timers };
}

function generation(text, seed, emotion = 'joy', backend = 'null') {
    return {
        success: true,
        image: `data:image/png;base64,${text}`,
        original_text: text,
        prompt: `Prompt for ${text}`,
        primary_emotion: emotion,
        confidence: 91,
        emotions: [{ name: emotion, score: 91 }],
        valence: emotion === 'joy' ? 0.8 : -0.7,
        arousal: 0.4,
        style: 'artistic',
        seed,
        backend,
        timestamp: '2026-10-05T00:00:00Z'
    };
}

test('history selection restores analysis, seed, image and report downloads', async () => {
    const app = createApp(async (_, options) => {
        const request = JSON.parse(options.body);
        return { json: async () => generation(request.text, request.seed,
            request.text === 'older' ? 'sadness' : 'joy', request.text === 'older' ? 'null' : 'hf-api') };
    });
    app.getElementById('emotionInput').value = 'older';
    app.getElementById('seedInput').value = '123';
    await app.api.generateImage();
    app.getElementById('emotionInput').value = 'latest';
    app.getElementById('seedInput').value = '456';
    await app.api.generateImage();
    assert.equal(app.getElementById('generationBackend').textContent, 'Hosted generation · hf-api backend');

    app.getElementById('historyThumbnails').children[1].click();

    assert.equal(app.getElementById('generatedImage').src, 'data:image/png;base64,older');
    assert.equal(app.getElementById('primaryEmotion').textContent, 'Sadness');
    assert.equal(app.getElementById('promptText').textContent, '"Prompt for older"');
    assert.equal(app.getElementById('userInputDisplay').textContent, 'older');
    assert.equal(app.getElementById('lastSeedValue').textContent, '123');
    assert.equal(app.getElementById('generationBackend').textContent, 'Test image · null backend');
    app.getElementById('copySeedBtn').click();
    app.api.downloadImage();
    app.api.downloadReport();
    assert.deepEqual(app.clipboard, ['123']);
    assert.equal(app.downloads[0].href, 'data:image/png;base64,older');
    assert.deepEqual(app.reportImages, ['data:image/png;base64,older']);
    assert(app.reportText.includes('Sadness'));
    assert(app.reportText.includes('"older"'));
    assert(!app.reportText.includes('"latest"'));
    assert.deepEqual(app.errors, []);
});

for (const [backend, label] of [
    ['null', 'Test image · null backend'],
    ['diffusers', 'Local generation · diffusers backend'],
    ['hf-api', 'Hosted generation · hf-api backend'],
    ['custom-generator', 'Image backend · custom-generator']
]) {
    test(`generation identifies the ${backend} backend after a neutral empty state`, async () => {
        const app = createApp(async () => ({ json: async () => generation('image', 1, 'joy', backend) }));
        assert.equal(app.getElementById('generationBackend').textContent, 'Image backend');
        app.getElementById('emotionInput').value = 'I feel happy';
        await app.api.generateImage();
        assert.equal(app.getElementById('generationBackend').textContent, label);
        assert.deepEqual(app.errors, []);
    });
}

test('superseded analysis cannot overwrite newer input or repopulate cleared input', async () => {
    const pending = [];
    // Deliberately ignore abort in the mock: late responses must also be discarded.
    const app = createApp((_, options) => new Promise(resolve => pending.push({ resolve, signal: options.signal })));
    const input = app.getElementById('emotionInput');
    input.value = 'I am very sad';
    const older = app.api.analyzeLive(input.value);
    input.value = 'I am very happy';
    const latest = app.api.analyzeLive(input.value);
    assert(pending[0].signal.aborted);
    pending[1].resolve({ json: async () => generation('happy', 1) });
    await latest;
    pending[0].resolve({ json: async () => generation('sad', 2, 'sadness') });
    await older;
    assert.equal(app.getElementById('liveStatusText').textContent, 'Joy');

    input.value = 'I am still sad';
    const cleared = app.api.analyzeLive(input.value);
    input.value = '';
    input.dispatch('input');
    assert(pending[2].signal.aborted);
    pending[2].resolve({ json: async () => generation('sad', 3, 'sadness') });
    await cleared;
    assert.equal(app.getElementById('liveStatusText').textContent, 'Ready');
    assert(app.getElementById('liveEmotionBars').classList.contains('muted'));
    assert.deepEqual(app.errors, []);
});

test('editing invalidates analysis before the debounce timer fires', async () => {
    let resolve;
    const app = createApp(() => new Promise(done => { resolve = done; }));
    const input = app.getElementById('emotionInput');
    input.value = 'I feel sad';
    const older = app.api.analyzeLive(input.value);
    input.value = 'I feel happy';
    input.dispatch('input');
    assert.equal(app.timers.size, 1);
    resolve({ json: async () => generation('sad', 3, 'sadness') });
    await older;
    assert.notEqual(app.getElementById('liveStatusText').textContent, 'Sadness');
    assert.deepEqual(app.errors, []);
});

test('unsafe seeds are rejected and the largest safe seed is submitted exactly', async () => {
    const requests = [];
    const app = createApp(async (_, options) => {
        const request = JSON.parse(options.body);
        requests.push(request);
        return { json: async () => generation(request.text, request.seed) };
    });
    app.getElementById('emotionInput').value = 'I feel happy';
    app.getElementById('seedInput').value = '9007199254740993';
    await app.api.generateImage();
    assert.equal(requests.length, 0);
    assert.equal(app.getElementById('errorText').textContent, 'Seed is too large to preserve exactly.');
    assert(app.getElementById('errorDetails').textContent.includes('9007199254740991'));
    app.getElementById('seedInput').value = '9007199254740991';
    await app.api.generateImage();
    assert.equal(requests[0].seed, Number.MAX_SAFE_INTEGER);
    assert.equal(app.getElementById('lastSeedValue').textContent, '9007199254740991');
    assert.deepEqual(app.errors, []);
});
