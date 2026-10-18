#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

FILES = {
    ".env.example": "GEMINI_API_KEY=\nGEMINI_TEXT_MODEL=gemini-3.8-flash\nGEMINI_IMAGE_MODEL=gemini-nano-banana-2.1\nGEMINI_VIDEO_MODEL=veo-3.1-generate-preview\nBUFFER_API_KEY=\nBUFFER_ORGANIZATION_ID=6ad3ea55ae4e83b03a8ee01d\nBUFFER_CHANNEL_ID=6ad3eff6778b240e2f3fae0d\nCLOUDINARY_URL=\nPUBLISH_MODE=live\nENABLE_IMAGE_GENERATION=true\nENABLE_VIDEO_GENERATION=true\n",
    ".gitignore": ".env\n__pycache__/\n*.py[cod]\n*$py.class\n.pytest_cache/\n.venv/\nvenv/\nassets/generated/\n*.tmp\ndeploy_to_github.py\n",
    "requirements.txt": "google-genai>=1.10.0\ncloudinary>=1.41.0\nPillow>=10.4.0\npydantic>=2.9.0\npython-dotenv>=1.0.1\nrequests>=2.32.3\npytest>=8.3.3\npyyaml>=6.0.2\n",
    "config/influencer.json": '{\n  "name": "Shasha Laurent",\n  "handle": "solace_shasha",\n  "apparent_age": 30,\n  "Fictional_Persona_Note": "Fictional AI-generated lifestyle and fitness persona.",\n  "biography": {\n    "marital_status": "Divorced, self-possessed, and thriving in her second chapter",\n    "financial_status": "Ultra-high-net-worth, financially independent investor and private equity exit founder",\n    "personality_traits": ["confident", "intelligent", "elegant", "disciplined", "sophisticated", "warm", "charismatic", "independent", "occasionally playful"],\n    "voice_and_tone": "Articulate, grounded, quietly confident, conversational, warm with subtle dry wit.",\n    "forbidden_claims": ["Real brand sponsorships or paid partnerships", "Medical, clinical, or dietitian credentials"]\n  },\n  "visual_identity": {\n    "facial_features": "High sculpted cheekbones, softly defined jawline, symmetrical oval face, warm hazel-green almond-shaped eyes",\n    "hair": "Rich espresso-chestnut brown hair, shoulder-blade length",\n    "skin_tone": "Sun-kissed warm olive complexion with natural skin texture",\n    "body_type": "Athletic, toned, statuesque 5\'8\" build",\n    "makeup_and_styling": \"Minimalist luxury 'no-makeup' makeup\",\n    "recurring_wardrobe": ["Monochrome sculpted athletic sets"],\n    "photography_style": "Shot on 35mm and 50mm full-frame prime lenses, f/2.0 aperture"\n  },\n  "reference_assets": {\n    "local_path": "assets/character_reference.png",\n    "public_url": ""\n  },\n  "content_pillars": {\n    "fitness_workout": {"target_ratio": 0.45, "description": "Gym workouts, strength training"},\n    "wellness_nutrition": {"target_ratio": 0.20, "description": "Balanced whole-food meals"},\n    "luxury_lifestyle": {"target_ratio": 0.20, "description": "Quiet-luxury fashion, architectural interiors"},\n    "mindset_independence": {"target_ratio": 0.15, "description": "Confidence, personal standards"}\n  }\n}\n',
    "state/posted_slots.json": '{\n  "character_reference": {"local_path": "assets/character_reference.png", "public_url": "", "generated_at": null},\n  "slots": {},\n  "recent_history": []\n}\n',
    "assets/.gitkeep": "",
    "app/__init__.py": '"""AI Influencer Automation Package."""\n',
    "app/config.py": 'from __future__ import annotations\nimport os\nfrom pathlib import Path\nfrom dataclasses import dataclass\nPROJECT_ROOT = Path(__file__).resolve().parent.parent\n',
    ".github/workflows/auto-post.yml": 'name: AI Influencer Daily Auto-Post\non:\n  schedule:\n    - cron: \'30 3 * * *\'\n    - cron: \'30 8 * * *\'\n    - cron: \'30 13 * * *\'\n  workflow_dispatch:\njobs:\n  run:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n'
}

def main():
    root = Path.cwd()
    for rel, content in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    print("[+] Project files written.")

if __name__ == "__main__":
    sys.exit(main())
