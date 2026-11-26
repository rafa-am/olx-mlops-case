# mlt — Model Lifecycle Toolkit: Architecture

Two views: the local MVP (what is implemented) and production on AWS (design only, not implemented).

---

## Local MVP

The CLI is a thin layer. All logic lives in `mlt.core`, which coordinates three interfaces: `DataSource`, `Runner`, and `RunStore`. Each has a local implementation in the MVP and a clear evolution path toward the cloud.

### Training flow

```mermaid
flowchart TD
    DS[Scientist] -->|mlt train| CLI[CLI\nmlt train]
    CLI --> CORE[mlt.core.trainer]

    CORE -->|1 static contract check| CONTRACT[contract.py\nAST only, no imports]
    CORE -->|2 compute fingerprint| FP[fingerprint.py\ncode + params + schema + deps + python + seed]
    FP -->|identical run exists| SKIP[return existing run_id]

    CORE -->|3 build env| RUNNER[Runner\nDocker or venv]
    RUNNER -->|deep validation inside env| WORKER_CHECK[_worker/train.py\n--validate-only]
    WORKER_CHECK -->|exit 3| NORUN[no run created]

    CORE -->|4 create run| STORE[RunStore\nfiles + SQLite]
    CORE -->|5 execute fit| WORKER[_worker/train.py\nno network, read-only data]
    WORKER -->|model, metrics.json, schema_effective.yaml| STORE
    STORE -->|6 write manifest atomically| MANIFEST[manifest.json\nsource of truth]
```

### Serving flow

```mermaid
flowchart TD
    DS[Scientist] -->|mlt serve run_id| CLI[CLI\nmlt serve]
    CLI -->|resolve run via RunStore| STORE[RunStore]
    STORE -->|run_dir| CLI

    CLI -->|determine runner from manifest| RUNNER{Runner}
    RUNNER -->|venv| VENV[~/.mlt/envs/env_hash/bin/python\n-m mlt.serving.server]
    RUNNER -->|docker| DOCKER[docker run\n-v run_dir:/run:ro\nmlt-env:hash]

    VENV --> SERVER[server.py\nloads model from run_dir/code/model.py]
    DOCKER --> SERVER

    SERVER -->|FastAPI app| APP[create_app\nmanifest, schema, model]
    APP -->|GET /health 200| CLI
    CLI -->|prints URL + curl example| DS

    CLIENT[Client] -->|POST /predict| APP
    APP -->|validate schema| SCHEMA[schema_effective.yaml\nfrom run, never from root]
    APP -->|model.predict| MODEL[MLModel instance]
    MODEL -->|predictions| CLIENT
```

### Component map

```mermaid
flowchart LR
    subgraph CLI [src/mlt/cli]
        TC[train_cmd] & SC[serve_cmd] & IC[init_cmd] & CC[check_cmd]
    end

    subgraph CORE [src/mlt/core]
        TR[trainer.py] & FP[fingerprint.py] & SEED[seed.py] & CONTR[contract.py]
    end

    subgraph MODEL [src/mlt/model]
        BASE[MLModel\nabstract] --> SKL[SklearnModel\njoblib + pandas]
    end

    subgraph RUNNERS [src/mlt/runners]
        VENV[VenvRunner] & DOCK[DockerRunner]
    end

    subgraph STORE [src/mlt/runstore]
        MAN[manifest.py] & ST[store.py\nfiles + SQLite]
    end

    subgraph SERVING [src/mlt/serving]
        APP[app.py\ncreate_app] & SRV[server.py\nuvicorn entry point]
    end

    CLI --> CORE
    CORE --> RUNNERS
    CORE --> STORE
    CORE --> MODEL
    CLI --> SERVING
    SERVING --> MODEL
    SERVING --> STORE
```

### Run directory structure

```
<project>/.mlt/runs/<run_id>/
├── manifest.json          source of truth (inputs, fingerprint, environment, outputs, error)
├── model/                 artifacts written by save() — format is the scientist's choice
│   └── model.joblib       (SklearnModel default)
├── metrics.json
├── train.log              stdout + stderr from fit()
├── params.yaml            effective params at training time
├── schema_effective.yaml  declared or inferred; read by the server at serve time
└── code/                  snapshot of the project (no data/, no .mlt/)
    ├── model.py
    └── requirements.txt
```

`verify` retrains from `code/` inside this directory, never from the project root. The run is immutable after `status: succeeded`.

---

## Production on AWS (design only)

The three MVP interfaces (`DataSource`, `Runner`, `RunStore`) map directly to AWS services. The CLI becomes a client of a control API; the scientist's workflow does not change.

```mermaid
flowchart LR
    DS[Scientist\nCLI mlt + Git] -->|push| GIT[(Git\nCodeCommit / GitHub)]
    GIT --> CB[CodeBuild\nbuild mlt-env image]
    CB -->|push image| ECR[(ECR\nmlt-env images)]

    DS -->|mlt train| API[Control API\nAPI Gateway + Lambda]
    API -->|submit job| TRAIN[Training\nAWS Batch or SageMaker Training\nno internet, VPC endpoints]
    ECR -->|environment image| TRAIN
    LH[(S3 + Iceberg\nlakehouse)] -->|DataSource: snapshot URI| TRAIN
    TRAIN -->|model, metrics, manifest| S3[(S3\nrun artifacts)]
    TRAIN -->|run metadata| DDB[(DynamoDB\nRunStore index)]

    DS -->|mlt serve run_id| API
    API -->|deploy task| SERVE[Inference\nECS Fargate + ALB]
    ECR -->|environment image| SERVE
    S3 -->|read-only model mount| SERVE

    DS -->|POST /predict| SERVE
```

### Evolution mapping

| MVP component | AWS equivalent | Notes |
|---------------|---------------|-------|
| `LocalDataSource` | `S3DataSource` with Iceberg snapshot | `--snapshot-data` becomes an immutable S3 path; `verify` always works |
| `VenvRunner` / `DockerRunner` | AWS Batch job or SageMaker Training job | Same `Runner` interface; job definition replaces Dockerfile |
| `RunStore` (files + SQLite) | S3 artifacts + DynamoDB index | Manifest stays the source of truth; DynamoDB replaces SQLite |
| `mlt serve` (foreground subprocess) | ECS Fargate task behind ALB | `serve --detach` maps to a task launch; `ps`/`stop` map to task describe/stop |
| CodeBuild | Triggered on Git push | Builds and pushes `mlt-env:<hash>` to ECR; Docker layer cache = same speed |

**What does not change:** the scientist's contract (`model.py`, `requirements.txt`, `params.yaml`, `schema.yaml`) and the manifest format. A run produced locally and one produced on AWS are identical JSON objects, and `diff`/`verify` work across both.

**Network isolation in production:** training jobs run in a VPC with no internet gateway; S3 and ECR are accessed via VPC endpoints. Same guarantee as `--network none` in the local Docker runner.
