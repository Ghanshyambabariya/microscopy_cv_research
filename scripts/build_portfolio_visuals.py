from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def write_svg(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def system_architecture_svg() -> str:
    return """<svg width="1200" height="620" viewBox="0 0 1200 620" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1200" y2="620" gradientUnits="userSpaceOnUse">
      <stop stop-color="#08111F"/>
      <stop offset="0.55" stop-color="#102A35"/>
      <stop offset="1" stop-color="#172416"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="12" stdDeviation="12" flood-color="#000000" flood-opacity="0.35"/>
    </filter>
    <style>
      .title{font:700 38px Segoe UI,Arial,sans-serif;fill:#F8FAFC}
      .subtitle{font:500 18px Segoe UI,Arial,sans-serif;fill:#C9D6E2}
      .h{font:700 20px Segoe UI,Arial,sans-serif;fill:#F8FAFC}
      .t{font:500 15px Segoe UI,Arial,sans-serif;fill:#D7E2EA}
      .tiny{font:600 13px Segoe UI,Arial,sans-serif;fill:#9ED0FF}
      .box{rx:22;filter:url(#shadow)}
      .arrow{stroke:#8BD7CE;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}
    </style>
  </defs>
  <rect width="1200" height="620" fill="url(#bg)"/>
  <circle cx="1040" cy="90" r="96" stroke="#6CE5D5" stroke-opacity="0.18" stroke-width="2"/>
  <circle cx="1120" cy="185" r="64" stroke="#F7C948" stroke-opacity="0.20" stroke-width="2"/>
  <text x="62" y="70" class="title">MicroForge AI Research Architecture</text>
  <text x="64" y="106" class="subtitle">A multimodal materials-intelligence stack for microscopy, process signals, tool wear, and property prediction.</text>

  <rect x="58" y="165" width="205" height="138" class="box" fill="#10233A" stroke="#3F83C6"/>
  <text x="84" y="204" class="h">Data Layer</text>
  <text x="84" y="235" class="t">SEM image registries</text>
  <text x="84" y="260" class="t">20 kHz force signals</text>
  <text x="84" y="285" class="t">Tool-wear tables</text>

  <rect x="345" y="165" width="220" height="138" class="box" fill="#112B31" stroke="#2B8C99"/>
  <text x="372" y="204" class="h">Preprocessing</text>
  <text x="372" y="235" class="t">schema checks</text>
  <text x="372" y="260" class="t">grouped splits</text>
  <text x="372" y="285" class="t">feature extraction</text>

  <rect x="650" y="165" width="220" height="138" class="box" fill="#2B2718" stroke="#B9902E"/>
  <text x="677" y="204" class="h">Model Zoo</text>
  <text x="677" y="235" class="t">UNet / DeepLab</text>
  <text x="677" y="260" class="t">MicroNet / Swin ready</text>
  <text x="677" y="285" class="t">RF / multitask heads</text>

  <rect x="955" y="165" width="185" height="138" class="box" fill="#202843" stroke="#6F8FE8"/>
  <text x="982" y="204" class="h">Evaluation</text>
  <text x="982" y="235" class="t">IoU / F1 / R2</text>
  <text x="982" y="260" class="t">leaderboards</text>
  <text x="982" y="285" class="t">visual reports</text>

  <path d="M266 234 H337" class="arrow"/>
  <path d="M568 234 H642" class="arrow"/>
  <path d="M873 234 H947" class="arrow"/>
  <path d="M327 400 C430 342 530 342 632 400" class="arrow" stroke-dasharray="8 8"/>
  <path d="M632 400 C745 462 845 462 958 400" class="arrow" stroke-dasharray="8 8"/>

  <rect x="130" y="395" width="235" height="110" class="box" fill="#0F1E31" stroke="#385C8C"/>
  <text x="158" y="433" class="h">Microscopy CV</text>
  <text x="158" y="463" class="t">segmentation, overlays, error maps</text>
  <text x="158" y="488" class="tiny">NASA EBC + SEM suite</text>

  <rect x="482" y="395" width="235" height="110" class="box" fill="#102D2D" stroke="#2B8C99"/>
  <text x="510" y="433" class="h">Active Learning</text>
  <text x="510" y="463" class="t">MC-dropout entropy acquisition</text>
  <text x="510" y="488" class="tiny">low-label microscopy loop</text>

  <rect x="835" y="395" width="235" height="110" class="box" fill="#30291A" stroke="#B9902E"/>
  <text x="863" y="433" class="h">Materials ML</text>
  <text x="863" y="463" class="t">signals, wear, properties</text>
  <text x="863" y="488" class="tiny">process-structure-property</text>
</svg>"""


def active_learning_svg() -> str:
    return """<svg width="1200" height="420" viewBox="0 0 1200 420" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg2" x1="0" y1="0" x2="1200" y2="420" gradientUnits="userSpaceOnUse">
      <stop stop-color="#0B1220"/>
      <stop offset="1" stop-color="#16302C"/>
    </linearGradient>
    <style>
      .title{font:700 34px Segoe UI,Arial,sans-serif;fill:#F8FAFC}
      .label{font:700 20px Segoe UI,Arial,sans-serif;fill:#F8FAFC}
      .text{font:500 15px Segoe UI,Arial,sans-serif;fill:#D7E2EA}
      .step{filter:drop-shadow(0 12px 18px rgba(0,0,0,.35))}
      .arrow{stroke:#6CE5D5;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}
    </style>
  </defs>
  <rect width="1200" height="420" rx="28" fill="url(#bg2)"/>
  <text x="54" y="65" class="title">Low-Label SEM Active Learning Loop</text>
  <g class="step">
    <rect x="62" y="145" width="180" height="120" rx="22" fill="#13243B" stroke="#4D89C8"/>
    <text x="91" y="188" class="label">Seed Labels</text>
    <text x="91" y="220" class="text">small expert set</text>
  </g>
  <g class="step">
    <rect x="305" y="145" width="180" height="120" rx="22" fill="#122D35" stroke="#2B8C99"/>
    <text x="333" y="188" class="label">Train UNet</text>
    <text x="333" y="220" class="text">class-weighted loss</text>
  </g>
  <g class="step">
    <rect x="548" y="145" width="205" height="120" rx="22" fill="#302B1A" stroke="#B9902E"/>
    <text x="578" y="188" class="label">Predict Pool</text>
    <text x="578" y="220" class="text">MC dropout entropy</text>
  </g>
  <g class="step">
    <rect x="817" y="145" width="170" height="120" rx="22" fill="#202843" stroke="#6F8FE8"/>
    <text x="847" y="188" class="label">Acquire</text>
    <text x="847" y="220" class="text">most uncertain</text>
  </g>
  <g class="step">
    <rect x="1045" y="145" width="105" height="120" rx="22" fill="#172E20" stroke="#66B879"/>
    <text x="1072" y="188" class="label">Report</text>
    <text x="1071" y="220" class="text">IoU/F1</text>
  </g>
  <path d="M245 205 H298" class="arrow"/>
  <path d="M488 205 H541" class="arrow"/>
  <path d="M756 205 H810" class="arrow"/>
  <path d="M990 205 H1038" class="arrow"/>
  <path d="M902 286 C780 360 472 360 392 286" class="arrow" stroke-dasharray="9 9"/>
  <text x="502" y="357" class="text">new labels are added, then the model is retrained and benchmarked again</text>
</svg>"""


def multimodal_cards_svg() -> str:
    return """<svg width="1200" height="470" viewBox="0 0 1200 470" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="1200" height="470" rx="30" fill="#F6F8F2"/>
  <style>
    .title{font:800 34px Segoe UI,Arial,sans-serif;fill:#132019}
    .h{font:800 22px Segoe UI,Arial,sans-serif;fill:#132019}
    .t{font:500 15px Segoe UI,Arial,sans-serif;fill:#34443A}
    .m{font:800 24px Segoe UI,Arial,sans-serif;fill:#0F5E5C}
  </style>
  <text x="54" y="62" class="title">Portfolio Evidence Map</text>
  <rect x="54" y="105" width="206" height="260" rx="24" fill="#E7F1F1" stroke="#5FA9A5"/>
  <text x="82" y="150" class="h">Microscopy CV</text>
  <text x="82" y="184" class="t">SEM segmentation</text>
  <text x="82" y="210" class="t">active learning</text>
  <text x="82" y="236" class="t">synthetic data</text>
  <text x="82" y="306" class="m">IoU 0.1174</text>
  <rect x="286" y="105" width="206" height="260" rx="24" fill="#EDF0FA" stroke="#859DE5"/>
  <text x="314" y="150" class="h">Signal ML</text>
  <text x="314" y="184" class="t">20 kHz force data</text>
  <text x="314" y="210" class="t">spectral features</text>
  <text x="314" y="236" class="t">quality regression</text>
  <text x="314" y="306" class="m">R2 0.9998</text>
  <rect x="518" y="105" width="206" height="260" rx="24" fill="#FFF4D8" stroke="#D3AA46"/>
  <text x="546" y="150" class="h">Tool Wear</text>
  <text x="546" y="184" class="t">real datasets</text>
  <text x="546" y="210" class="t">grouped splits</text>
  <text x="546" y="236" class="t">wear stages</text>
  <text x="546" y="306" class="m">R2 0.8680</text>
  <rect x="750" y="105" width="206" height="260" rx="24" fill="#EAF6E7" stroke="#7EBD6E"/>
  <text x="778" y="150" class="h">Property ML</text>
  <text x="778" y="184" class="t">composition table</text>
  <text x="778" y="210" class="t">feature importance</text>
  <text x="778" y="236" class="t">strength prediction</text>
  <text x="778" y="306" class="m">R2 0.8990</text>
  <rect x="982" y="105" width="164" height="260" rx="24" fill="#EFE9F6" stroke="#A98BD0"/>
  <text x="1010" y="150" class="h">Platform</text>
  <text x="1010" y="184" class="t">configs</text>
  <text x="1010" y="210" class="t">reports</text>
  <text x="1010" y="236" class="t">tests</text>
  <text x="1010" y="306" class="m">11 tests</text>
</svg>"""


def main() -> None:
    write_svg(ASSETS / "system_architecture.svg", system_architecture_svg())
    write_svg(ASSETS / "active_learning_loop.svg", active_learning_svg())
    write_svg(ASSETS / "portfolio_evidence_map.svg", multimodal_cards_svg())
    print("wrote portfolio visuals to assets/")


if __name__ == "__main__":
    main()
