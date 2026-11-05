# Model and provider configuration

Copy `reward_engine/configs/model_config.example.json` to a private path and replace every `YOUR_MODEL_ID`. Both benchmark `prepare` commands accept `--model-config FILE` and repeatable `--role-model ROLE=MODEL`. No personal provider address or model combination is selected for a new campaign.

The profile uses `reward_engine.model_config.v1`. File settings apply first, CLI role overrides second, and protocol-derived attack roles last. Three-B attack seats inherit the corresponding jury model and token limit. Hypothesis roles are unused in three-B. Missing required models fail preparation. Derived attack roles cannot be overridden independently.

Optional fields include `role_max_tokens`, `model_aliases`, `role_request_options_by_model`, and `concurrency`. Office additionally accepts `native_grader`, `measurement_candidate_concurrency`, and `delivery_review_json_schema_models`. Native grading has a separate route and never supplies learning or selection inputs. Claw selects its native grader with `--native-grader-model`.

Provider settings use CLI > environment > selected dotenv. Set `--backend-url` or `REWARD_ENGINE_BACKEND_URL`; an API address is required. `OPENAI_API_KEY` is the default credential variable. Use `--key-env CUSTOM_API_KEY` or `REWARD_ENGINE_KEY_ENV` for another name. A matching `CUSTOM_BASE_URL` is also supported. Select private dotenv files with `--credentials FILE` or `REWARD_ENGINE_CREDENTIALS_FILE`. Never commit credentials. URLs containing embedded credentials are rejected.

Preparation binds original profile bytes and final effective settings into configuration identity and manifest. Runtime checks detect changed bytes and settings. Existing frozen runs remain unchanged; settings cannot be changed through `run`. Keep their original frozen implementation and evidence when reproducing them.
