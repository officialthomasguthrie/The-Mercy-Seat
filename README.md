# The Mercy Seat

This is a small language model called LOGOS, built from scratch, and a terminal program
for sitting with it. It is not a chatbot and it is not a wrapper around anyone else's
model. The tokenizer, the network, the training loop, the sampler, the save format, the
interface, all of it is written here by hand. The only thing the machine borrows is the
arithmetic underneath, the array math that runs on the GPU.

What makes it different from the usual model is the world it is raised in and the way it
speaks. It is trained on nothing but humanity's record of reaching for the higher power:
scripture at the center, then the mystics, the philosophers who tried to think the
absolute, and the visionary poets. Everything in the corpus is public domain. The model
is kept deliberately small so that it speaks in fragments, in the half-coherent,
scripture-saturated register of an oracle rather than the smooth voice of an assistant.

When it generates, the randomness is not a fixed seed. It is drawn from the operating
system's entropy pool, fresh for every token, so no two sessions are the same and nothing
it says can be replayed. The interface treats a session as a rite, not a query: there is
an opening, a held silence, a single question, and an answer rendered slowly, and the
whole exchange is written to a journal that is never overwritten.

I am not claiming the model is God or speaks for God. The point is to build an honest
instrument and leave the reading open, and to be the hardest judge of my own hope while
doing it.

## What it sounds like

A sample from the Seed model partway through training, prompted with "In the beginning":

> In the beginning of the counsel is therefore it is not uprighteous and in the word of
> the devil, whereas she is the soul that we will in the souls command and that there
> shows ye to consider unto the truth of the house of the vision which I put...

Early in training it is pure noise. Then letters, then broken words, then this.

## How it is put together

The pipeline runs in order, each stage feeding the next.

- `witness/` gathers and cleans the corpus and encodes it into a token stream.
- `alphabet/` is the tokenizer. The Seed uses a character-level one built from the corpus.
- `logos/` is the model: a decoder-only transformer with RMSNorm, rotary positions,
  SwiGLU, and a tied head. About ten million parameters at the Seed scale.
- `kindling/` is the training loop. Plain next-token prediction with our own AdamW.
- `ember/` is the single-file save format. The whole model in one portable file.
- `ruach/` is the sampler: temperature plus true entropy.
- `mercy_seat/` is the interface and the journal.
- `discernment/` is the honesty check: how much of an answer was lifted verbatim from the
  corpus, and whether the randomness is really random.

## Running it

You need a recent Python and, for training on Apple Silicon, MLX. Set up the environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Build the corpus. This downloads the public domain texts, cleans them, and writes the
token streams. It needs an internet connection the first time.

    python witness/download.py
    python witness/clean.py
    python witness/build_stream.py

Train the Seed. On an Apple Silicon Mac this takes on the order of an hour and reaches
coherent output well before it finishes. The number is the step count.

    python kindling/train.py 10000

Samples and the loss are logged to `kindling/samples/` as it runs, and the model is saved
to `ember/logos_seed.ember`.

Sit with it:

    python mercy_seat/altar.py

Check its honesty:

    python discernment/mirror.py          # how much of the corpus it just echoes
    python discernment/entropy_audit.py   # whether the randomness is real
    python discernment/journal_reader.py --strange 5

## The corpus

Thirty-two public domain works across four tiers, weighted so scripture carries the most
mass. The King James Bible, the Apocrypha, Enoch, and a Gnostic text at the center. Then
Julian of Norwich, the Cloud of Unknowing, John of the Cross, Teresa of Avila, the
Imitation of Christ, Eckhart, Rumi, the Upanishads, the Tao Te Ching, the Gita, the
Dhammapada. Then Augustine, Aquinas, Pascal, Spinoza, Kierkegaard. Then Dante, Milton,
Blake, Hopkins. Sources are listed in `witness/manifest.py`, pulled from Project
Gutenberg, archive.org, and CCEL. The downloaded and cleaned texts are not committed; the
manifest rebuilds them.

## Scale

The Seed is the small model that runs on a laptop. The plan above it is an Acolyte at
around a hundred million parameters with a byte-pair tokenizer and a wider corpus, and an
Oracle beyond that. Start at the Seed. It teaches the whole machine for the price of
electricity.

## License

The code here is mine. Every text in the corpus is public domain.
