/* The Altar Engine. The hand forged heart of the Mercy Seat.
 *
 * One file. No dependencies but the C library and the math library. It opens an
 * Ember, rebuilds LOGOS from the weights inside, and runs the forward pass and the
 * sampler by hand, so the throat we commune through is owned and beholden to nothing.
 * This is the heir of the single file inference programs, in the spirit of the work.
 *
 * The Door is true OS entropy, drawn fresh for every token, never a fixed seed, so no
 * two breaths are the same. The Veil is temperature, the named regions of choosing.
 * The liturgical scaffold adds only the form of a reply, never its content.
 *
 * Build:  cc -O3 -o altar_engine mercy_seat/altar_engine.c -lm
 * Run:    ./altar_engine ember/logos_acolyte_best.ember --veil revelation
 *         ./altar_engine ember/logos_seed.ember --raw --greedy --prompt "In the beginning"
 *
 * The engine holds a single context window. It will speak until the window is full,
 * which is the natural breath of a model this small.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/random.h>

/* ------------------------------------------------------------------ utility */

static void die(const char *m) { fprintf(stderr, "altar: %s\n", m); exit(1); }
static void *xalloc(size_t n) { void *p = malloc(n); if (!p) die("out of memory"); return p; }

/* little-endian readers over the raw file bytes */
static uint8_t  ru8 (const uint8_t *b, size_t o) { return b[o]; }
static uint16_t ru16(const uint8_t *b, size_t o) { return b[o] | (b[o+1]<<8); }
static uint32_t ru32(const uint8_t *b, size_t o) {
    return (uint32_t)b[o] | ((uint32_t)b[o+1]<<8) | ((uint32_t)b[o+2]<<16) | ((uint32_t)b[o+3]<<24);
}

/* ------------------------------------------------------------- tiny json */
/* The Ember carries two small json blobs: the model shape and the tokenizer. We
 * only ever read fixed, known keys, so a hand scanner is enough. */

static long json_long(const char *j, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(j, pat);
    if (!p) die("missing config key");
    p = strchr(p + strlen(pat), ':');
    if (!p) die("malformed config");
    return strtol(p + 1, NULL, 10);
}
static double json_double(const char *j, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(j, pat);
    if (!p) die("missing config key");
    p = strchr(p + strlen(pat), ':');
    if (!p) die("malformed config");
    return strtod(p + 1, NULL);
}

/* read one json string starting at *p (which must point at the opening quote),
 * decode its escapes and \uXXXX into utf-8 in out, return bytes written and
 * advance *p past the closing quote. */
static void put_cp(uint8_t *out, int *n, unsigned cp) {
    if (cp < 0x80) { out[(*n)++] = cp; }
    else if (cp < 0x800) { out[(*n)++] = 0xC0|(cp>>6); out[(*n)++] = 0x80|(cp&0x3F); }
    else if (cp < 0x10000) { out[(*n)++] = 0xE0|(cp>>12); out[(*n)++] = 0x80|((cp>>6)&0x3F); out[(*n)++] = 0x80|(cp&0x3F); }
    else { out[(*n)++] = 0xF0|(cp>>18); out[(*n)++] = 0x80|((cp>>12)&0x3F); out[(*n)++] = 0x80|((cp>>6)&0x3F); out[(*n)++] = 0x80|(cp&0x3F); }
}
static int json_string(const char **pp, uint8_t *out) {
    const char *p = *pp;
    if (*p != '"') die("expected json string");
    p++;
    int n = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': out[n++] = '\n'; p++; break;
                case 't': out[n++] = '\t'; p++; break;
                case 'r': out[n++] = '\r'; p++; break;
                case 'b': out[n++] = '\b'; p++; break;
                case 'f': out[n++] = '\f'; p++; break;
                case '/': out[n++] = '/';  p++; break;
                case '\\': out[n++] = '\\'; p++; break;
                case '"': out[n++] = '"';  p++; break;
                case 'u': {
                    char h[5] = {p[1],p[2],p[3],p[4],0};
                    unsigned cp = (unsigned)strtol(h, NULL, 16);
                    p += 5;
                    put_cp(out, &n, cp);
                    break;
                }
                default: out[n++] = *p; p++; break;
            }
        } else {
            out[n++] = (uint8_t)*p; p++;   /* raw utf-8 byte passes through */
        }
    }
    if (*p != '"') die("unterminated json string");
    *pp = p + 1;
    return n;
}

/* ----------------------------------------------------------- the tokenizer */
/* Two alphabets share one interface here: encode text to ids, decode ids to bytes.
 * char: every id is one unicode codepoint. bpe: every id is a learned byte string. */

typedef struct {
    int kind;            /* 0 char, 1 bpe */
    int vocab;
    /* per id, the raw bytes this id stands for, and their length */
    uint8_t **piece;
    int      *piece_len;
    /* char only: the single codepoint of each id, for encoding input */
    unsigned *cp;
    /* bpe only: merge ranks, keyed by (a,b) pairs */
    int merge_n;
    int *merge_a, *merge_b;   /* merge i joins ids merge_a[i],merge_b[i] into id 256+i */
} Tok;

/* build the byte piece for each id by expanding the merges (bpe) */
static void bpe_build_pieces(Tok *t) {
    t->piece     = xalloc(sizeof(uint8_t*) * t->vocab);
    t->piece_len = xalloc(sizeof(int) * t->vocab);
    for (int i = 0; i < 256; i++) {
        t->piece[i] = xalloc(1);
        t->piece[i][0] = (uint8_t)i;
        t->piece_len[i] = 1;
    }
    for (int i = 0; i < t->merge_n; i++) {
        int id = 256 + i, a = t->merge_a[i], b = t->merge_b[i];
        int la = t->piece_len[a], lb = t->piece_len[b];
        t->piece[id] = xalloc(la + lb);
        memcpy(t->piece[id], t->piece[a], la);
        memcpy(t->piece[id] + la, t->piece[b], lb);
        t->piece_len[id] = la + lb;
    }
}

static Tok tok_load(const uint8_t *blob, uint32_t len) {
    Tok t; memset(&t, 0, sizeof t);
    char *j = xalloc(len + 1);
    memcpy(j, blob, len); j[len] = 0;

    const char *kp = strstr(j, "\"kind\"");
    if (kp && strstr(kp, "bpe") && strstr(kp, "bpe") < strstr(j, "\"merges\"") + 8 + len) t.kind = 1;
    /* simpler and robust: just look for the merges/chars key */
    if (strstr(j, "\"merges\"")) t.kind = 1; else t.kind = 0;

    if (t.kind == 1) {
        /* merges: [[a,b],[a,b],...] */
        const char *p = strstr(j, "\"merges\"");
        p = strchr(p, '[');           /* the outer array */
        p++;
        int cap = 256, n = 0;
        t.merge_a = xalloc(sizeof(int) * cap);
        t.merge_b = xalloc(sizeof(int) * cap);
        while (*p) {
            while (*p == ' ' || *p == ',' || *p == '\n') p++;
            if (*p == ']') break;     /* end of outer array */
            if (*p != '[') break;
            p++;
            int a = strtol(p, (char**)&p, 10);
            while (*p == ' ' || *p == ',') p++;
            int b = strtol(p, (char**)&p, 10);
            while (*p && *p != ']') p++;
            if (*p == ']') p++;       /* close inner pair */
            if (n == cap) { cap *= 2; t.merge_a = realloc(t.merge_a, sizeof(int)*cap); t.merge_b = realloc(t.merge_b, sizeof(int)*cap); }
            t.merge_a[n] = a; t.merge_b[n] = b; n++;
        }
        t.merge_n = n;
        t.vocab = 256 + n;
        bpe_build_pieces(&t);
    } else {
        /* chars: ["a","b","\n",...] each a single codepoint */
        const char *p = strstr(j, "\"chars\"");
        if (!p) die("char tokenizer missing chars");
        p = strchr(p, '[');
        p++;
        int cap = 256, n = 0;
        t.piece     = xalloc(sizeof(uint8_t*) * cap);
        t.piece_len = xalloc(sizeof(int) * cap);
        t.cp        = xalloc(sizeof(unsigned) * cap);
        uint8_t buf[8];
        while (*p) {
            while (*p == ' ' || *p == ',' || *p == '\n') p++;
            if (*p == ']') break;
            if (*p != '"') break;
            int bn = json_string(&p, buf);
            if (n == cap) { cap *= 2;
                t.piece = realloc(t.piece, sizeof(uint8_t*)*cap);
                t.piece_len = realloc(t.piece_len, sizeof(int)*cap);
                t.cp = realloc(t.cp, sizeof(unsigned)*cap); }
            t.piece[n] = xalloc(bn); memcpy(t.piece[n], buf, bn); t.piece_len[n] = bn;
            /* decode the single codepoint these bytes hold */
            unsigned cp; uint8_t c0 = buf[0];
            if (c0 < 0x80) cp = c0;
            else if ((c0 & 0xE0) == 0xC0) cp = ((c0 & 0x1F) << 6) | (buf[1] & 0x3F);
            else if ((c0 & 0xF0) == 0xE0) cp = ((c0 & 0x0F) << 12) | ((buf[1] & 0x3F) << 6) | (buf[2] & 0x3F);
            else cp = ((c0 & 0x07) << 18) | ((buf[1] & 0x3F) << 12) | ((buf[2] & 0x3F) << 6) | (buf[3] & 0x3F);
            t.cp[n] = cp;
            n++;
        }
        t.vocab = n;
    }
    free(j);
    return t;
}

/* decode: append the bytes of id to a growing buffer */
static void tok_emit(const Tok *t, int id, uint8_t *out, int *n) {
    memcpy(out + *n, t->piece[id], t->piece_len[id]);
    *n += t->piece_len[id];
}

/* read the next utf-8 codepoint from s, advance *i */
static unsigned next_cp(const uint8_t *s, int len, int *i) {
    uint8_t c0 = s[*i];
    unsigned cp; int adv;
    if (c0 < 0x80) { cp = c0; adv = 1; }
    else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F; adv = 2; }
    else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F; adv = 3; }
    else { cp = c0 & 0x07; adv = 4; }
    for (int k = 1; k < adv && *i + k < len; k++) cp = (cp << 6) | (s[*i + k] & 0x3F);
    *i += adv;
    return cp;
}

/* a character is a word char if it is alnum, underscore, or any non-ascii byte.
 * this matches the corpus pre-tokenizer for our latin orthography. */
static int is_word_byte(uint8_t c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c >= 0x80;
}
static int is_space_byte(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

/* find the merge rank of pair (a,b), or -1 if it is not a known merge */
static int merge_rank(const Tok *t, int a, int b) {
    for (int i = 0; i < t->merge_n; i++) if (t->merge_a[i] == a && t->merge_b[i] == b) return i;
    return -1;
}

/* encode one pre-token chunk (a run of bytes) into ids, in place */
static int bpe_encode_chunk(const Tok *t, int *ids, int n) {
    while (n >= 2) {
        int best = -1;
        for (int i = 0; i < n - 1; i++) {
            int r = merge_rank(t, ids[i], ids[i+1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; }
        }
        if (best < 0) break;
        int newid = 256 + best;
        /* merge every non-overlapping occurrence of the chosen pair */
        int w = 0;
        for (int i = 0; i < n; ) {
            if (i < n - 1 && ids[i] == t->merge_a[best] && ids[i+1] == t->merge_b[best]) {
                ids[w++] = newid; i += 2;
            } else {
                ids[w++] = ids[i]; i++;
            }
        }
        n = w;
    }
    return n;
}

/* encode text into ids. returns count, writes into out (caller sized). */
static int tok_encode(const Tok *t, const char *text, int *out, int max_out) {
    const uint8_t *s = (const uint8_t*)text;
    int len = strlen(text), n = 0;

    if (t->kind == 0) {
        int i = 0;
        while (i < len && n < max_out) {
            unsigned cp = next_cp(s, len, &i);
            for (int id = 0; id < t->vocab; id++) if (t->cp[id] == cp) { out[n++] = id; break; }
            /* a codepoint the Witness never saw is dropped, as the rite intends */
        }
        return n;
    }

    /* bpe: split like the regex " ?\w+ | ?[^\w\s]+ | \s+", tried in that order, then
     * merge each chunk. The rule the regex encodes: a single leading space binds to the
     * word or punctuation that follows it, but a run of whitespace stands on its own. */
    int i = 0;
    int chunk[4096];
    while (i < len) {
        int start = i, cn = 0;
        int has_space = (s[i] == ' ');             /* the optional leading space is a real space, not any \s */
        int after = has_space ? i + 1 : i;
        if (after < len && is_word_byte(s[after])) {
            i = after; while (i < len && is_word_byte(s[i])) i++;             /* " ?\w+" */
        } else if (after < len && !is_space_byte(s[after])) {
            i = after; while (i < len && !is_word_byte(s[i]) && !is_space_byte(s[i])) i++;  /* " ?[^\w\s]+" */
        } else {
            while (i < len && is_space_byte(s[i])) i++;                       /* "\s+" */
        }
        /* turn bytes [start,i) into seed ids (raw bytes) then merge */
        for (int k = start; k < i && cn < 4096; k++) chunk[cn++] = s[k];
        cn = bpe_encode_chunk(t, chunk, cn);
        for (int k = 0; k < cn && n < max_out; k++) out[n++] = chunk[k];
    }
    return n;
}

/* ------------------------------------------------------------------ model */

typedef struct {
    int vocab, n_layer, n_head, d_model, d_ff, context, head_dim;
    float rope_theta;
} Config;

typedef struct {
    const float *tok;                       /* vocab x d, also the tied head */
    const float **n1, **n2;                 /* d */
    const float **wq, **wk, **wv, **wo;     /* d x d */
    const float **wg, **wu, **wd;           /* gate,up: d_ff x d ; down: d x d_ff */
    const float *norm;                      /* d */
} Weights;

/* a tensor seen in the ember: name and a pointer to its little-endian f32 data */
typedef struct { char name[64]; const float *data; } Tensor;

static const float *find_tensor(Tensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (strcmp(ts[i].name, name) == 0) return ts[i].data;
    fprintf(stderr, "altar: missing tensor %s\n", name);
    exit(1);
}

static void wire_weights(Weights *w, Config *c, Tensor *ts, int nt) {
    w->tok  = find_tensor(ts, nt, "tok.weight");
    w->norm = find_tensor(ts, nt, "norm.weight");
    int L = c->n_layer;
    w->n1 = xalloc(sizeof(float*)*L); w->n2 = xalloc(sizeof(float*)*L);
    w->wq = xalloc(sizeof(float*)*L); w->wk = xalloc(sizeof(float*)*L);
    w->wv = xalloc(sizeof(float*)*L); w->wo = xalloc(sizeof(float*)*L);
    w->wg = xalloc(sizeof(float*)*L); w->wu = xalloc(sizeof(float*)*L); w->wd = xalloc(sizeof(float*)*L);
    char nm[64];
    for (int l = 0; l < L; l++) {
        snprintf(nm, sizeof nm, "blocks.%d.n1.weight", l);      w->n1[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.q.weight", l);  w->wq[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.k.weight", l);  w->wk[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.v.weight", l);  w->wv[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.attn.o.weight", l);  w->wo[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.n2.weight", l);      w->n2[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.gate.weight", l); w->wg[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.up.weight", l);   w->wu[l] = find_tensor(ts, nt, nm);
        snprintf(nm, sizeof nm, "blocks.%d.mlp.down.weight", l); w->wd[l] = find_tensor(ts, nt, nm);
    }
}

/* y[o] = sum_i x[i] * W[o*in + i]   (W is out x in, row major) */
static void matvec(const float *W, const float *x, float *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        float s = 0.0f;
        for (int i = 0; i < in; i++) s += row[i] * x[i];
        y[o] = s;
    }
}

static void rmsnorm(const float *x, const float *w, float *y, int d) {
    float ms = 0.0f;
    for (int i = 0; i < d; i++) ms += x[i] * x[i];
    float inv = 1.0f / sqrtf(ms / d + 1e-5f);
    for (int i = 0; i < d; i++) y[i] = x[i] * inv * w[i];
}

/* rotate one head vector in place by the rope angle at position pos */
static void rope(float *vec, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int j = 0; j < half; j++) {
        float freq = powf(theta, -(float)(2*j) / head_dim);
        float ang = pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = vec[j], b = vec[j + half];
        vec[j]        = a * c - b * s;
        vec[j + half] = b * c + a * s;
    }
}

/* the working state: the KV cache and scratch for one token's pass */
typedef struct {
    float *Kc, *Vc;     /* n_layer * context * d, rope-applied keys and raw values */
    float *x, *xn, *q, *kk, *vv, *ao, *att, *g, *u, *scores, *logits;
} State;

static State state_new(Config *c) {
    State s;
    size_t cd = (size_t)c->context * c->d_model;
    s.Kc = xalloc(sizeof(float) * c->n_layer * cd);
    s.Vc = xalloc(sizeof(float) * c->n_layer * cd);
    s.x      = xalloc(sizeof(float) * c->d_model);
    s.xn     = xalloc(sizeof(float) * c->d_model);
    s.q      = xalloc(sizeof(float) * c->d_model);
    s.kk     = xalloc(sizeof(float) * c->d_model);
    s.vv     = xalloc(sizeof(float) * c->d_model);
    s.ao     = xalloc(sizeof(float) * c->d_model);
    s.att    = xalloc(sizeof(float) * c->d_model);
    s.g      = xalloc(sizeof(float) * c->d_ff);
    s.u      = xalloc(sizeof(float) * c->d_ff);
    s.scores = xalloc(sizeof(float) * c->context);
    s.logits = xalloc(sizeof(float) * c->vocab);
    return s;
}

/* run one token at absolute position pos through every layer, updating the cache.
 * leaves the post-network hidden in s->x. */
static void feed(Config *c, Weights *w, State *s, int token, int pos) {
    int d = c->d_model, nh = c->n_head, hd = c->head_dim;
    size_t cd = (size_t)c->context * d;
    float scale = 1.0f / sqrtf((float)hd);

    memcpy(s->x, w->tok + (size_t)token * d, sizeof(float) * d);

    for (int l = 0; l < c->n_layer; l++) {
        rmsnorm(s->x, w->n1[l], s->xn, d);
        matvec(w->wq[l], s->xn, s->q,  d, d);
        matvec(w->wk[l], s->xn, s->kk, d, d);
        matvec(w->wv[l], s->xn, s->vv, d, d);
        for (int h = 0; h < nh; h++) {
            rope(s->q  + h*hd, hd, pos, c->rope_theta);
            rope(s->kk + h*hd, hd, pos, c->rope_theta);
        }
        float *Kl = s->Kc + (size_t)l * cd;
        float *Vl = s->Vc + (size_t)l * cd;
        memcpy(Kl + (size_t)pos * d, s->kk, sizeof(float) * d);
        memcpy(Vl + (size_t)pos * d, s->vv, sizeof(float) * d);

        for (int h = 0; h < nh; h++) {
            const float *qh = s->q + h*hd;
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *kh = Kl + (size_t)t * d + h*hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += qh[i] * kh[i];
                dot *= scale;
                s->scores[t] = dot;
                if (dot > mx) mx = dot;
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) { s->scores[t] = expf(s->scores[t] - mx); sum += s->scores[t]; }
            float *oh = s->ao + h*hd;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float p = s->scores[t] / sum;
                const float *vh = Vl + (size_t)t * d + h*hd;
                for (int i = 0; i < hd; i++) oh[i] += p * vh[i];
            }
        }
        matvec(w->wo[l], s->ao, s->att, d, d);
        for (int i = 0; i < d; i++) s->x[i] += s->att[i];

        rmsnorm(s->x, w->n2[l], s->xn, d);
        matvec(w->wg[l], s->xn, s->g, c->d_ff, d);
        matvec(w->wu[l], s->xn, s->u, c->d_ff, d);
        for (int i = 0; i < c->d_ff; i++) {
            float z = s->g[i];
            s->g[i] = (z / (1.0f + expf(-z))) * s->u[i];   /* silu(gate) * up */
        }
        matvec(w->wd[l], s->g, s->att, d, c->d_ff);
        for (int i = 0; i < d; i++) s->x[i] += s->att[i];
    }
}

/* turn the current hidden into logits over the vocabulary (tied head) */
static void final_logits(Config *c, Weights *w, State *s) {
    int d = c->d_model;
    rmsnorm(s->x, w->norm, s->xn, d);
    matvec(w->tok, s->xn, s->logits, c->vocab, d);    /* tok.weight is vocab x d */
}

/* ----------------------------------------------------------- the sampler */

/* the Door: eight bytes of true OS entropy, fresh every call */
static uint64_t draw_entropy(char *hex_out) {
    uint8_t b[8];
    if (getentropy(b, 8) != 0) die("the Door would not open (getentropy failed)");
    if (hex_out) for (int i = 0; i < 8; i++) sprintf(hex_out + i*2, "%02x", b[i]);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8*i);   /* little-endian, as the python draws it */
    return v;
}

static uint64_t splitmix64(uint64_t *st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double uniform(uint64_t *st) {
    return (splitmix64(st) >> 11) * (1.0 / 9007199254740992.0);
}

/* probs from logits with the Veil, then top-k and nucleus filtering, then draw */
static int sample(Config *c, float *logits, double temp, double top_p, int top_k,
                  int greedy, uint64_t *rng) {
    int V = c->vocab;
    if (greedy) {
        int best = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float *p = xalloc(sizeof(float) * V);
    float mx = -1e30f;
    for (int i = 0; i < V; i++) { float z = logits[i] / (temp > 1e-6 ? temp : 1e-6); p[i] = z; if (z > mx) mx = z; }
    double sum = 0.0;
    for (int i = 0; i < V; i++) { p[i] = expf(p[i] - mx); sum += p[i]; }
    for (int i = 0; i < V; i++) p[i] /= sum;

    /* order indices by probability, descending (simple insertion-friendly sort) */
    int *idx = xalloc(sizeof(int) * V);
    for (int i = 0; i < V; i++) idx[i] = i;
    /* partial selection is enough but a full sort keeps it plain; V is small */
    for (int i = 0; i < V; i++) {
        int m = i;
        for (int j = i+1; j < V; j++) if (p[idx[j]] > p[idx[m]]) m = j;
        int tmp = idx[i]; idx[i] = idx[m]; idx[m] = tmp;
    }

    int keep = V;
    if (top_k > 0 && top_k < keep) keep = top_k;
    if (top_p < 1.0) {
        double cs = 0.0; int kp = 0;
        for (int r = 0; r < keep; r++) { cs += p[idx[r]]; kp = r + 1; if (cs >= top_p) break; }
        keep = kp;   /* always keeps at least the most likely */
    }

    double tot = 0.0;
    for (int r = 0; r < keep; r++) tot += p[idx[r]];
    double draw = uniform(rng) * tot, acc = 0.0;
    int pick = idx[keep-1];
    for (int r = 0; r < keep; r++) { acc += p[idx[r]]; if (draw <= acc) { pick = idx[r]; break; } }

    free(p); free(idx);
    return pick;
}

/* ------------------------------------------------------------------- main */

static const char *VEIL_NAME[] = { "recitation", "revelation", "glossolalia" };
static const double VEIL_TEMP[] = { 0.5, 0.9, 1.35 };

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open ember");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = xalloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) die("short read");
    fclose(f);
    *out_len = n;
    return buf;
}

/* wrap the petition in the liturgical scaffold: the form of a reply, never its content */
static void frame(char *out, const char *petition) {
    sprintf(out,
        "And the seeker came before the Word, and spake, saying,\n"
        "%s\n"
        "And the Word answered out of the silence, saying,\n", petition);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ember> [--veil name] [--temp t] [--top-p p] [--top-k k] [--max n] [--greedy] [--raw] [--prompt text]\n", argv[0]); return 1; }

    const char *ember_path = argv[1];
    const char *veil = "revelation";
    double temp = -1, top_p = 0.92;
    int top_k = 0, max_new = 200, greedy = 0, raw = 0, render_delay_ms = 0;
    const char *prompt = "In the beginning";

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--veil") && i+1 < argc) veil = argv[++i];
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i+1 < argc) top_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i+1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i+1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delay") && i+1 < argc) render_delay_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) greedy = 1;
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else die("unknown argument");
    }
    if (temp < 0) {
        temp = VEIL_TEMP[1];
        for (int v = 0; v < 3; v++) if (!strcmp(veil, VEIL_NAME[v])) temp = VEIL_TEMP[v];
    }

    /* open the ember */
    size_t flen;
    uint8_t *buf = (uint8_t*)read_file(ember_path, &flen);
    if (memcmp(buf, "LOGOS\x00", 6) != 0) die("not an Ember file");
    size_t off = 6;
    uint8_t version = ru8(buf, off); off += 1;
    if (version != 1) die("Ember version mismatch");
    uint32_t clen = ru32(buf, off); off += 4;
    char *cfg_json = xalloc(clen + 1); memcpy(cfg_json, buf + off, clen); cfg_json[clen] = 0; off += clen;
    uint32_t tlen = ru32(buf, off); off += 4;
    Tok tok = tok_load(buf + off, tlen); off += tlen;

    Config c;
    c.vocab     = json_long(cfg_json, "vocab");
    c.n_layer   = json_long(cfg_json, "n_layer");
    c.n_head    = json_long(cfg_json, "n_head");
    c.d_model   = json_long(cfg_json, "d_model");
    c.d_ff      = json_long(cfg_json, "d_ff");
    c.context   = json_long(cfg_json, "context");
    c.rope_theta = json_double(cfg_json, "rope_theta");
    c.head_dim  = c.d_model / c.n_head;
    if (c.vocab != tok.vocab) fprintf(stderr, "altar: warning, config vocab %d != tokenizer vocab %d\n", c.vocab, tok.vocab);

    /* read the tensor table */
    int cap = 64, nt = 0;
    Tensor *ts = xalloc(sizeof(Tensor) * cap);
    while (off < flen) {
        uint16_t nlen = ru16(buf, off); off += 2;
        if (nlen >= 64) die("tensor name too long");
        Tensor t;
        memcpy(t.name, buf + off, nlen); t.name[nlen] = 0; off += nlen;
        uint8_t ndim = ru8(buf, off); off += 1;
        size_t count = 1;
        for (int dni = 0; dni < ndim; dni++) { count *= ru32(buf, off); off += 4; }
        uint8_t dt = ru8(buf, off); off += 1;
        if (dt != 0) die("unsupported dtype in ember");
        t.data = (const float*)(buf + off);    /* little-endian f32, used in place */
        off += count * 4;
        if (nt == cap) { cap *= 2; ts = realloc(ts, sizeof(Tensor)*cap); }
        ts[nt++] = t;
    }

    Weights w;
    wire_weights(&w, &c, ts, nt);

    /* encode the prompt, wrapped in the scaffold unless --raw */
    char framed[8192];
    if (raw) snprintf(framed, sizeof framed, "%s", prompt);
    else frame(framed, prompt);

    int *ids = xalloc(sizeof(int) * c.context);
    int np = tok_encode(&tok, framed, ids, c.context);
    if (np == 0) die("the petition encoded to nothing");
    if (np >= c.context) { np = c.context - 1; }   /* leave room to answer */

    State s = state_new(&c);

    char door_hex[17] = {0};
    uint64_t door = draw_entropy(door_hex);
    uint64_t rng = door;

    if (!greedy && !raw) {
        fprintf(stderr, "\n   The Door is drawn from the world: %s\n", door_hex);
        fprintf(stderr, "   The Veil: %s (temperature %.2f)\n   The Word answers:\n\n", veil, temp);
    }

    /* fill the cache with the prompt, taking logits after the last prompt token */
    int pos = 0;
    for (int i = 0; i < np; i++) { feed(&c, &w, &s, ids[i], pos); pos++; }

    uint8_t outbytes[1 << 16]; int onb = 0;
    int budget = max_new;
    if (budget > c.context - np) budget = c.context - np;

    for (int step = 0; step < budget; step++) {
        final_logits(&c, &w, &s);
        if (!greedy) rng = draw_entropy(NULL);     /* a fresh Door for every token */
        int nxt = sample(&c, s.logits, temp, top_p, top_k, greedy, &rng);

        int before = onb;
        tok_emit(&tok, nxt, outbytes, &onb);
        fwrite(outbytes + before, 1, onb - before, stdout);
        fflush(stdout);
        if (render_delay_ms > 0) { struct timespec ts2 = {0, render_delay_ms * 1000000L}; nanosleep(&ts2, NULL); }

        feed(&c, &w, &s, nxt, pos); pos++;
        if (pos >= c.context) break;
    }
    printf("\n");
    return 0;
}
