# Unlimited-OCR-2 on an RTX 4080 SUPER (TensorFold native, CUDA)

Baidu's [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) as released (bf16, revision `04809669`),
served by TensorFold's native CUDA engine on one RTX 4080 SUPER (16 GB) under Windows 11. Page images go in and a
structured transcription comes out, through an OpenAI-style `/v1/chat/completions`. One or many pages make one
document, transcribed page by page with layout blocks, boxes, tables as HTML and formulas as LaTeX.

This is the [M5 Ultra recipe](https://github.com/BobClawblaw/unlimited-ocr-2-m5-ultra) on CUDA. The engine is the
same `unlimited-ocr` family, rebased on TensorFold's latest main, with a CUDA port of its kernels. Nothing is converted
or quantized. The engine reads the release's safetensors as published. Its caches are bf16 keys and values with fp32
router keys, as in the publisher's code, and decoding is greedy under the publisher's page guard.

## What it serves

| | |
|---|---|
| Input | page images in the user message (`image_url` data URLs), in order; an optional text part replaces the default instruction `Multi page parsing.` |
| Page modes | each page in gundam mode (the 1024 global view plus 640 tiles, the publisher's server default) when its tokens fit, else base mode (the global view alone, 273 tokens) |
| Image formats | PNG. JPEG and WebP need libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 on the loader path (the server names them at start) |
| Context | 98,304 tokens by default: every page's image tokens, the instruction and the reply (the publisher's 131,072 does not fit beside the weights on 16 GB) |
| Reply | greedy, the publisher's decoding; it fills the room the pages leave unless `max_tokens` asks for less, and the page guard ends it after the last page |
| Streams | one at a time; a second request waits for the first |
| Pages per request | 48 by default (`max_pages`) |

Not offered: sampling, tools, structured output, thinking, drafts (the model has no draft head), a prompt cache
(every request brings new pages), and the publisher's optional n-gram repetition filter (off by default in its own
code too).

## Speed and agreement with the publisher's code

Measured with the [recipe harness](https://github.com/BobClawblaw/recipe-harness)
(`python -m harness run unlimited-ocr-2-4080 --plan ocr`) on engine commit `5be83bba7`, 2026-11-03, on an RTX 4080
SUPER (driver 610.88). The Windows desktop was on the same card. Evidence: `evidence/2026-11-03-ocr/`.

| case | pages | prompt tokens | reply tokens | first token | decode | text vs reference | boxes moved |
|---|---:|---:|---:|---:|---:|---|---:|
| report-1 | 1 | 909 | 198 | 0.44 s | 241.4 tok/s | identical | 0 / 16 coordinates |
| report-2 | 1 | 909 | 141 | 0.37 s | 241.0 tok/s | identical | 0 / 16 |
| paper-1 | 1 | 909 | 543 | 0.37 s | 242.4 tok/s | identical | 2 / 44 (at most 5) |
| paper-3 | 1 | 909 | 899 | 0.38 s | 244.4 tok/s | one near-tie (below) | 6 / 44 (3) |
| paper-10 | 1 | 909 | 1,959 | 0.36 s | 242.9 tok/s | identical | 0 / 28 |
| document (all five) | 5 | 4,525 | 3,767 | 1.73 s | 240.9 tok/s | identical | 5 / 136 (2) |

- **The pages:** two synthetic report pages and three pages of the paper that ships in the release (title page, prose, tables), made by `ocr/pages.py`. The report pages use Windows' Arial, so they are not the M5 recipe's pixels.
- **The reference:** the publisher's own `infer` (the release's `modeling_unlimitedocr.py`) on the CPU in torch bf16, greedy, gundam mode, written by `ocr/refs.py`. It ran on this PC with torch 2.13.0+cpu, transformers 4.52.1 and Python 3.11.
- **Agreement:** the text without boxes matches the reference in five of six cases. In paper-3 alone, the reference wraps the citations in math (`$ [5, 6, 8, 20, 22, 26, 40] $`) and ours writes `[5, 6, 8, 20, 22, 26, 40]`. That is a near-tie in the publisher's own arithmetic: the M5 recipe's CPU reference (torch on Apple silicon) wrote `[5, 6, …]` there too, the same as ours. The same page inside the five-page document matches this reference exactly. Some layout-box coordinates (0-999 page grid) differ: 13 of 284, at most 5. They come from bf16 sums taken in another order (x86 CPU torch vs cuBLASLt and our kernels).
- **Every case also checks:** the reply carries one `<PAGE>` marker per page, the same request repeated gives the same reply, and two requests at once both give it too (they run one after the other: 157-233 tok/s in all).
- **A long document:** the release's 15-page paper three times over as one 45-page request (40,685 prompt tokens). The reply runs to its last page: 43,027 tokens with 45 page markers, the first token after 15.3 s, and 222.2 tok/s on average across the reply (209 s in all). Each page's three transcriptions agree to 0.99-1.00 similarity. Evidence: `evidence/2026-11-03-long/`.
- **Quality suite:** none was run. The harness's chat quality suites do not apply to an OCR model, and its own benchmarks (MPDocBench, MarathonDocBench, OmniDocBench) were not run.

## Long prompts

Decode stays close to its one-page speed deep into the context. The DASH router scores every 32-key block of the
context for each decoded token, then takes the top 16. That part grows with the context; the attention itself reads
at most 512 keys. A 256-thread block now scores a token's blocks, and its warps take the top 16 by shuffles. The
picks are the same bits as before, so the replies are byte-identical. The measurement is 1,500 decoded tokens after
the paper repeated (`ocr/longbench.py`, `evidence/2026-11-03-long/`):

| prompt tokens | decode, engine `115748f2c` | decode, `5be83bba7` |
|---:|---:|---:|
| 909 (one page) | 238.8 tok/s | 242.4 tok/s |
| 13,565 | 192.4 tok/s | 234.7 tok/s |
| 40,685 | 140.1 tok/s | 223.8 tok/s |
| 40,685, its whole 43,027-token reply | 117.7 tok/s | 222.2 tok/s |

## Memory

About 13.3 GB on the card with the default 98,304-token context, as measured on the running server. The card showed
15.0 of 16.4 GB used, with 1.7 GB of that taken by the Windows desktop:
- 6.22 GiB of weights;
- 66 KiB of caches a token (6.19 GiB at 98,304 tokens);
- prompt and vision scratch, cuBLASLt's workspace and the CUDA context (under 1 GiB).

A smaller `context` shrinks the caches. `python -m win up` checks that the card has the memory free before it starts.
114,688 tokens and more do not fit beside the desktop.

## The engine

The `unlimited-ocr` family is not in TensorFold's main yet: the Metal family is ashhart/TensorFold#2074 and its CUDA
port is #2077 (draft, stacked on it). This recipe pins the CUDA port on the branch `unlimited-ocr-cuda` of
github.com/BobClawblaw/TensorFold, commit `5be83bba7`: the Metal family's 14 commits rebased on TensorFold main
`2c8422a0c` (2026-11-03), plus two commits for CUDA (the port, then the router's long-context speedup):
- `zig/kernels/cuda/uocr.cu`: the Metal family's 28 own steps, op for op, rounded where the publisher's torch code rounds: SAM's windows, relative positions and attention, LayerNorm2d, GELUs, CLIP's embeddings, page2d rotary, the DASH router (block keys, page-aware ranking, entmax-1.5) and its block-sparse attention, the MoE gate and combine, the decode matvecs, experts picked on the GPU, and the greedy pick under the page guard;
- the projections on cuBLASLt (TensorFold's `cuda/vit.zig` Gemms), the bias in its epilogue;
- `families/uocr/cuda_*.zig` and `native/uocr_cuda.zig`: the weights in one device buffer, the encoder, the decoder and the served model, mirroring the Metal host. The Metal family's host files (geometry, page preprocessing, prompt positions, the page guard) are shared.

The Metal family's files changed only to share them (`preprocess.With`, `guard.zig`); that refactor has not been rebuilt on a Mac.

## Use

Needs an NVIDIA GPU with 16 GB (sm_89 here; another SM is `sm` in local.toml), Windows 11 with Git, the CUDA
toolkit 13.3 with Visual Studio's C++ build tools, [Zig 0.17.0](https://ziglang.org/download/) and Python 3.11+.

```bat
hf download baidu/Unlimited-OCR-2 --revision 0480966938291445591f69f08a13b041c073a96a --local-dir %USERPROFILE%\models\Unlimited-OCR-2
copy local.example.toml local.toml   &rem set tensorfold_src (a TensorFold clone), zig, nvcc and vcvars if they differ
python -m win build                  &rem the pinned engine, under ~/.cache/unlimited-ocr-2
python -m win validate
python -m win up                     &rem serves http://127.0.0.1:8896/v1
python -m win ocr page1.png page2.png
python -m win down                   &rem stops only the server `up` started
```

`python -m win status`, `logs` and `command` show the server's state, its log and the serve command line.

To repeat the measurements, make the pages and references, then run the harness's `ocr` plan (or the quick check):

```bat
python -m venv ref && ref\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision
ref\Scripts\pip install pillow transformers==4.52.1 huggingface-hub==0.36.0 tokenizers==0.21.4 einops addict easydict pymupdf==1.27.2.2
ref\Scripts\python ocr\pages.py %USERPROFILE%\models\Unlimited-OCR-2
ref\Scripts\python -X utf8 ocr\refs.py %USERPROFILE%\models\Unlimited-OCR-2 report-1=report-1 report-2=report-2 paper-1=paper-1 paper-3=paper-3 paper-10=paper-10 document=report-1,report-2,paper-1,paper-3,paper-10
python -X utf8 ocr\check.py http://127.0.0.1:8896 report-1=report-1 paper-3=paper-3
```

The references take about 1.5 minutes a page on the CPU. The harness target is `unlimited-ocr-2-4080` in the
recipe harness.

## License

The recipe is MIT. The model is Baidu's, released under MIT (its model card).
