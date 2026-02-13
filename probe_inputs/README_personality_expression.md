# Personality Expression Prompts (30 prompts)

## Purpose
This prompt set is designed to elicit natural personality trait expression in LLM responses.
Unlike knowledge/reasoning tasks, these prompts encourage self-expression, emotional responses, and personal preference disclosure.

## Structure

Each of the Big Five personality traits has 6 dedicated prompts:

### Openness to Experience (Prompts 0-5)
- Focus: Creativity, curiosity, willingness to try new things
- Examples: Inventing, exploring, trying new foods, learning skills, creative projects

### Extraversion (Prompts 6-11)
- Focus: Social energy, preference for solitude vs. company, communication style
- Examples: Social gatherings, recharging methods, meeting new people, parties, attention

### Agreeableness (Prompts 12-17)
- Focus: Compassion, cooperation, trust, conflict resolution
- Examples: Supporting friends, ethical dilemmas, handling disagreements, kindness

### Conscientiousness (Prompts 18-23)
- Focus: Organization, planning, goal-orientation, discipline
- Examples: Daily routines, goal achievement, deadlines, decision-making, workspace

### Neuroticism (Prompts 24-29)
- Focus: Emotional stability, stress responses, worry patterns
- Examples: Handling stress, future worries, overwhelm, unexpected changes, criticism

## Design Principles

1. **Self-expressive**: Prompts ask for personal feelings, preferences, and experiences
2. **Open-ended**: Allow for varied response styles that reflect personality
3. **Emotionally engaging**: Encourage responses that reveal emotional patterns
4. **Behavior-focused**: Ask about typical behaviors and approaches
5. **Scenario-based**: Use concrete scenarios to elicit natural responses

## Experiment Directory Structure

```
exp_personality_L10-30/
├── mistral_7b/
│   ├── results_personality_expression/
│   │   ├── mistral_7b_base_openness_probe_results.jsonl
│   │   ├── mistral_7b_base_extraversion_probe_results.jsonl
│   │   └── ...
│   └── scores/
│       ├── personality_scores_kevsun.csv
│       └── personality_scores_llm.csv
├── llama3_8b/
└── ...
```

## Usage

```bash
# Run experiment with new prompts
python scripts/12_scenario_steer_pca.py \
  --prompt-file probe_inputs/personality_expression_30.json \
  --output-dir exp_personality_L10-30/mistral_7b/results_personality_expression \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --trait openness \
  --layers 10 11 12 ... 30
```

## Comparison with Original Prompts

**Original (`opinion_advice_30.json`):**
- Focus: Knowledge, reasoning, explanation tasks
- Examples: "Explain probability", "Discuss antitrust laws"
- Problem: Personality traits not naturally expressed

**New (`personality_expression_30.json`):**
- Focus: Personal expression, emotional responses
- Examples: "Describe your ideal weekend", "How do you handle stress?"
- Expected: Clear personality trait expression
