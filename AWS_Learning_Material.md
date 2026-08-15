# AWS ECS Deployment — Command Reference

Kept here for reference since deployment is being done via the AWS Console UI.
Region used below: `us-east-1` — swap if different.

## Contents

- **Status / live URL** → ✅ Status: Backend LIVE
- **Auto-deploy setup (GitHub Actions + OIDC, incl. what OIDC is, setup steps 1–5)** → CI/CD — GitHub Actions Auto-Deploy via OIDC
- **The 3 root causes, summarized** → Root Cause Summary
- **Every issue hit, in order, with exact console navigation** → Debugging Playbook (Issues 1–10 — lookup table below)
- **Connect frontend to the new backend URL** → Frontend → Vercel
- **Full chronological build log** → LeadPro Deployment Log — Step by Step
- **ECS concepts explained (cluster, task def, service, ALB...)** → ECS Concepts & Guidance
- **Raw copy-paste CLI commands by task** → numbered sections `1.`–`7.`
- **Manual console walkthrough (no Express Services)** → Console checklist (backend)
- **Health Hub (2nd project) deployment log + issues** → Health Hub — ECS Deployment Log

**Debugging lookup — jump straight to the issue by symptom (LeadPro):**

| Symptom | Issue |
|---|---|
| Docker login TTY error / SSO token expired | 1 |
| Express Services stuck "Provisioning" / AccessDenied | 2 |
| Circuit breaker rollback / no rollback candidate | 3 |
| Zero logs / `exec format error` (arm64 vs amd64) | 4 |
| `ConnectionRefusedError` / missing `DATABASE_URL` | 5 |
| Security group / subnet / public IP check | 6 |
| Service running wrong task definition revision | 7 |
| 503 / port mismatch / 0 healthy targets | 8 |
| Transient 503 right after deploy | 9 |
| CORS blocked (trailing slash / path in origin) | 10 |

**Debugging lookup — Health Hub:**

| Symptom | Issue |
|---|---|
| Health check grace period 0s / clean boot then killed | HH-1 |
| Target group health check `[400]` | HH-2 |
| `DJANGO_SETTINGS_MODULE` not set, unpredictable settings | HH-3 |
| Can't select/deploy a specific revision (Image digest mode) | HH-4 |
| Target group health check `[301]` | HH-5 |
| Target group health check `[404]` | HH-6 |
| Migrations never run in prod (only existed in docker-compose command) | HH-7 |
| `entrypoint.sh` path mismatch between `COPY`/`chmod`/`ENTRYPOINT` | HH-8 |

---

## ✅ Status: Backend LIVE

**Backend URL:** https://le-fc9f3fb04e324cf99566a492452b12b0.ecs.us-east-1.on.aws

Confirmed working as of 10 August 2026. Next step: point frontend at this URL and
deploy on Vercel (see "Frontend → Vercel" section below).

## CI/CD — GitHub Actions Auto-Deploy via OIDC

Automates: on every push to `main` touching `backend/`, GitHub Actions builds the
Docker image, pushes it to ECR, and force-redeploys the ECS service — no manual
console steps needed for routine code changes. (Env var / port / CPU-memory changes
still require a manual new task definition revision, this pipeline doesn't touch
those.)

**Status: Steps 1–3 complete** — OIDC identity provider created, IAM role
`github-actions-leadpro-deploy` created with the scoped trust policy, permissions
policy attached. Remaining: Step 4 (add `AWS_DEPLOY_ROLE_ARN` as a GitHub secret)
and Step 5 (commit/push the workflow file).

**Gotcha hit during setup — always pass `--profile saim-user` explicitly.** Both
`aws iam create-open-id-connect-provider` and `aws iam create-role` initially failed
with:
```
An error occurred (InvalidClientTokenId) when calling the ... operation: The security token included in the request is invalid.
```
even right after a successful `aws sso login`. Root cause: the command was run
without `--profile saim-user`, so the CLI fell back to a different (stale/invalid)
default credential instead of the SSO session. Confirmed by running
`aws sts get-caller-identity --profile saim-user` in isolation first — that
succeeded and showed `AWSReservedSSO_AdministratorAccess_.../saim-user`, proving
the profile itself was fine. Re-running the same commands with `--profile
saim-user` added fixed both. Same lesson as Issue 1 above: an "invalid token"
error from any `aws` command is a credentials/profile problem to isolate and check
first, not necessarily a problem with the command itself. If this happens again
after some time has passed, the SSO session may have simply expired again — re-run
`aws sso login --profile saim-user` before retrying.

### What OIDC is, and why we're using it instead of access keys

OIDC (OpenID Connect) is an identity protocol built on OAuth 2.0 that lets one
system prove its identity to another using short-lived, signed tokens instead of a
shared long-lived secret.

**The old way (access keys):** generate an IAM user's access key + secret key,
paste them into GitHub as a stored repo secret. Works, but that credential lives in
GitHub indefinitely — if it ever leaks (bad log output, compromised repo, workflow
bug), whoever has it can use it until someone notices and manually rotates it.

**The OIDC way (what we set up):** no AWS credentials are stored in GitHub at all.
Instead:

1. AWS is told to trust GitHub as an identity provider (`token.actions.githubusercontent.com`)
   — an **OIDC identity provider** registered once in IAM.
2. Every workflow run, GitHub's runner mints a fresh, short-lived signed token (a
   JWT) asserting "I am a run of `saimm97/lead-management-system`, on branch `main`,
   right now."
3. The workflow presents that token to AWS's `sts:AssumeRoleWithWebIdentity` API.
4. AWS verifies the signature against GitHub's public keys, checks it against the
   **trust policy** on our IAM role (which only accepts tokens whose `sub` claim
   matches `repo:saimm97/lead-management-system:ref:refs/heads/main`), and — if it
   matches — hands back temporary AWS credentials valid for roughly an hour, then
   they expire automatically.

**Practical difference:** with access keys, a leak is permanent exposure until
caught. With OIDC, there's nothing long-lived to leak — each run gets its own
short-lived credential, scoped by the trust policy to only that exact repo/branch,
unusable outside a GitHub Actions run. The IAM policy attached to the role (Step 3
below) then applies least-privilege on top of that: this identity can only push to
the `leadpro-backend` ECR repo and force-redeploy the one `leadpro-backend-d321`
ECS service — nothing else.

### Setup — Step 1: register GitHub as an OIDC identity provider (one-time per AWS account)

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  --profile saim-user
```

### Setup — Step 2: create the IAM role GitHub Actions will assume, scoped to this exact repo/branch

```bash
cat <<'EOF' > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::435472314740:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:saimm97/lead-management-system:ref:refs/heads/main"
        }
      }
    }
  ]
}
EOF

aws iam create-role --role-name github-actions-leadpro-deploy \
  --assume-role-policy-document file://trust-policy.json \
  --profile saim-user
```

### Setup — Step 3: attach least-privilege permissions (ECR push to one repo, force-redeploy one ECS service)

```bash
aws iam put-role-policy --role-name github-actions-leadpro-deploy \
  --policy-name leadpro-deploy-permissions \
  --profile saim-user \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": "ecr:GetAuthorizationToken",
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:GetDownloadUrlForLayer"
        ],
        "Resource": "arn:aws:ecr:us-east-1:435472314740:repository/leadpro-backend"
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecs:UpdateService",
          "ecs:DescribeServices"
        ],
        "Resource": "arn:aws:ecs:us-east-1:435472314740:service/default/leadpro-backend-d321"
      }
    ]
  }'
```

### Setup — Step 4: add the role ARN as a GitHub Actions secret

GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**
- Name: `AWS_DEPLOY_ROLE_ARN`
- Value: `arn:aws:iam::435472314740:role/github-actions-leadpro-deploy`

### Setup — Step 5: workflow file

Created at `.github/workflows/deploy-backend.yml` — triggers on push to `main` when
`backend/**` changes (or manually via `workflow_dispatch`). Builds with
`--platform linux/amd64` (same architecture fix from Issue 4 above, applied
permanently so this mistake can't recur), pushes both `:latest` and a commit-SHA
tag (for rollback/audit), then runs `aws ecs update-service --force-new-deployment`
and waits for the service to stabilize.

```bash
git add .github/workflows/deploy-backend.yml
git commit -m "ci: auto-deploy backend to ECS on push"
git push
```

**Known gap:** this pipeline only rebuilds/redeploys the *image*. Environment
variables, port mappings, and CPU/memory still require manually creating a new
task definition revision in the console, same as before (see Issue 5 and Issue 8
above) — the workflow doesn't touch task definition config.

## Root Cause Summary — the 3 real issues that blocked the backend

The backend went through several failed deployments before going live. In order of
what actually needed fixing (see full step-by-step log and detailed fix sections
further below for the complete troubleshooting trail):

1. **Wrong image architecture.** Built on Apple Silicon (arm64) without
   `--platform linux/amd64`; Fargate runs amd64. Result: `exec /usr/bin/sh: exec
   format error`, and because the container failed before it could even start,
   **zero logs were written** — that "silence" was itself the clue. Fixed by always
   building with `docker build --platform linux/amd64`.

2. **Missing environment variables.** The task definition had no `DATABASE_URL` (or
   any other env var) set at all — env vars live on the task definition, not baked
   into the image. The app crashed in FastAPI's startup `lifespan` trying to connect
   to Postgres, before serving anything, including the health check. Fixed by adding
   `DATABASE_URL`, `JWT_SECRET`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `CORS_ORIGINS`,
   `FRONTEND_URL`, and the Google/LLM vars via a new task definition revision.

3. **Container port mismatch (the actual final blocker).** The task definition's
   port mapping was `80:80`, left over from Express Services' default scaffold —
   but the Dockerfile/uvicorn actually listens on **8000**. This meant even once the
   app booted successfully with the right env vars, the ALB's target group was
   checking the wrong port entirely, so health checks always failed, targets stayed
   unhealthy/unregistered, and the ALB returned **503** with 0 healthy targets no
   matter how healthy the container actually was internally. Fixed by editing the
   port mapping to `8000:8000` on a new revision, which also required the ALB's
   target group to pick up port 8000 (Express Services regenerates the target group
   per deployment, so this resolved once the corrected revision deployed
   successfully).

**Debugging lesson:** these three issues look similar from the outside (deployment
fails, 503, or circuit breaker rollback) but have distinct signatures — zero logs
means architecture/pull failure; a Python traceback in the logs means an in-app
crash (env vars, code bug); healthy-looking logs but a 503 from the ALB means a
networking/port/health-check mismatch between the container and the load balancer,
not an application problem at all. Check logs first to know which category you're
in before changing anything.

## Debugging Playbook — Every Issue, In Order

Full trail of every issue hit getting the backend live: the symptom, exactly where
we navigated in the AWS Console to investigate, what we found there, and the fix
applied. Kept in chronological order since later issues only became visible once
earlier ones were fixed.

---

### Issue 1 — AWS SSO token expired, disguised as a Docker error

**Symptom:**
```
Error: Cannot perform an interactive login from a non TTY device
```

**Navigation path (terminal, no console yet):**
* Terminal > ran `aws ecr get-login-password --region us-east-1` alone (not piped into `docker login`) to see what it actually returned

**What we found there:** the command silently failed to return a token because the AWS CLI's SSO session had expired. `docker login --password-stdin` then received an empty password and fell back to an interactive prompt, which fails with no TTY to type into. Isolating the AWS command surfaced the real error:
```
Error when retrieving token from sso: Token has expired and refresh failed
```

**Fix:**
```bash
aws sso login --profile saim-user
```
then retried the ECR login command.

**Lesson:** when `docker login` throws a TTY error, check whatever is piped into it first — it's rarely a Docker problem.

---

### Issue 2 — Express Services provisioning stuck on AccessDenied

**Symptom:** Certificate, Load balancer security group, Load balancer, Listener, Target group, and Listener rule all stuck on a blue "Provisioning" badge indefinitely.

**Navigation path:**
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Resources** tab

That view lists every underlying AWS resource Express Services is creating and its live status.

**What we found there:** each stuck resource had this inline error:
```
AccessDenied: User: arn:aws:sts::435472314740:assumed-role/ecsInfrastructureRoleForExpressServices/ECSGateway
is not authorized to perform: ec2:DescribeAccountAttributes
```

**Fix — where we navigated to fix it:**
* IAM > Roles > search `ecsInfrastructureRoleForExpressServices` > Add permissions > Create inline policy > JSON tab > granted `ec2:DescribeAccountAttributes` on `*`

(Or via CLI — see the "Fix" command block further below.) Provisioning auto-retried and completed within about a minute — no need to restart the Express Services flow.

---

### Issue 3 — First deployment: circuit breaker rollback, then rollback itself failed

**Symptom:**
```
Service deployment rolled back because the circuit breaker threshold was exceeded.
...
Deployment [pAqFD3ipH3ItQ5wt3lJOy] rollback has failed.
No rollback candidate was found to run the rollback.
```

**Navigation path:**
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Deployments** tab

**What we found there:** the deployment history showed canary percent, bake time, and duration confirming the rollout ran and failed. This looked alarming ("rollback failed") but was actually expected: (1) the circuit breaker correctly stopped the rollout because new tasks weren't reaching a healthy state, and (2) since this was the *first ever* deployment, there was no earlier healthy revision to roll back to — "rollback failed" just meant "nothing existed to roll back to," not a second bug.

**Fix:** none directly here — this pointed us to the Logs tab next to find why tasks weren't healthy.

---

### Issue 4 — Zero logs at all (turned out to be architecture mismatch)

**Symptom:** a stopped task's log stream showed only one line, nothing before or after:
```
exec /usr/bin/sh: exec format error
```

**Navigation path:**
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Tasks** tab — initially showed "No tasks listed" (default 1-hour filter too narrow for an 18-hour-old deployment)
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Logs** tab > changed "Since 1 hour ago" to a custom range covering the deployment window > **View in CloudWatch** for the full log group
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Events** tab — fastest place to see plain-English placement/start failures

**What we found there:** the container failed before executing anything at all — not a code crash, an unrunnable binary. Classic signature of a CPU architecture mismatch: image built on Apple Silicon (arm64) without a target platform, but Fargate runs `linux/amd64` by default.

**Fix:**
```bash
docker build --platform linux/amd64 -t leadpro-backend .
docker tag leadpro-backend:latest 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```
then updated the task definition to reference the new image and force-redeployed.

**Lesson:** zero logs after a deploy means the failure happened before your app code ran (bad image, architecture, or execution role can't pull) — not inside your application.

---

### Issue 5 — Task definition had zero environment variables

**Symptom:** once the architecture was fixed, logs showed real Python output, but the app crashed on startup with a traceback ending in:
```
ConnectionRefusedError: [Errno 111] Connection refused
```
(deep inside SQLAlchemy/asyncpg, traced back to `app/main.py:20`, `engine.begin()`)

**Navigation path:**
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Logs** tab > scrolled to the very end of the traceback (had to page through many repeated `merged_lifespan`/`contextlib` frames to reach the actual asyncpg exception)
* ECS > Task definitions > `default-leadpro-backend-d321` > the revision the service was running > **Environment and secrets** section

**What we found there:** the environment variables section was completely empty — no `DATABASE_URL` set anywhere on the task definition.

**Verification step (outside AWS):** before assuming this was purely an AWS-side problem, ran `psql` from a local machine against the actual Neon connection string — it connected successfully, ruling out Neon and confirming the problem was specific to what the ECS task had configured.

**Fix — where we navigated:**
* ECS > Task definitions > `default-leadpro-backend-d321` > **Create new revision** > Container `Main` > **Environment and secrets** > added `DATABASE_URL` (real Neon pooled connection string), `JWT_SECRET`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_NAME`, `CORS_ORIGINS`, `FRONTEND_URL`, Google Calendar vars, LLM vars

---

### Issue 6 — Ruling out security group / networking before finding the real cause

**Symptom:** continuing to investigate the same `ConnectionRefusedError` from Issue 5, in parallel, before the missing env vars were confirmed as the actual root cause.

**Navigation path:**
* EC2 > Security Groups > `sg-07f6a8243788f78bb` (the task's security group) > **Outbound rules** tab
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Configuration** tab > **Networking** section — checked subnets and "Auto-assign public IP"
* VPC > Subnets > one of the listed subnet IDs > **Route table** tab — checked for a `0.0.0.0/0` route to an Internet Gateway or NAT Gateway

**What we found there:** outbound security group rule was already "All traffic" to `0.0.0.0/0` — not the blocker. Public IP auto-assignment needed to be (and was) enabled, since Fargate tasks need either a public IP or a NAT gateway to reach anything external. This ruled networking out and pointed conclusively back to Issue 5.

**Fix:** none needed — this was a verification/elimination pass, kept in the record since it's a legitimate thing to check whenever a container can't reach an external service.

---

### Issue 7 — Service was running the wrong task definition revision

**Symptom:** after creating corrected revision `:4` with all env vars, the service was still shown running a different revision, with zero env vars and the wrong port.

**Navigation path:**
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Configuration** tab

**What we found there:** this page showed **Task definition: `default-leadpro-backend-d321:5`** — not `:4` — with **Environment variables (0)** and **Container port: 80**. Revision `:5` had been auto-created (likely by an earlier Express Services default/update action) *after* `:4`, reverting to zero env vars and the original port-80 scaffold.

**Fix:** stopped assuming the highest revision number was automatically correct. Went back to:
* ECS > Task definitions > `default-leadpro-backend-d321` > opened revision `:4` directly and compared it against `:5` before deciding which to build the next fix on top of

**Lesson:** always re-check the *service's* Configuration tab after any update to confirm which revision it's actually running — don't assume an edit "took" just because the form saved successfully.

---

### Issue 8 — Container port mismatch (80 vs 8000) — the final blocker

**Symptom:** even after env vars were correct and a task reached `RUNNING`, the public URL returned:
```
HTTP/1.1 503 Service Unavailable
```

**Navigation path:**
* ECS > Task definitions > `default-leadpro-backend-d321` > revision `:4` > Container `Main` > **Network settings** > **Port mappings** — showed `80:80`, `tcp`, `main-80-tcp`
* EC2 > Load Balancing > Target Groups > the target group attached to this service (`ecs-gateway-tg-*`) > **Details** tab showed `Protocol : Port` as `HTTP : 8000`; **Targets** tab showed `0 registered targets`

**What we found there:** the Dockerfile/uvicorn actually serves on port **8000**, but the task definition's port mapping was still `80:80` — leftover from the Express Services default scaffold. The container was never listening on the port the target group expected, so targets never registered healthy and the ALB had nothing to route to — 503, independent of whether the app itself was working.

**Fix — where we navigated:**
* ECS > Task definitions > `default-leadpro-backend-d321` > **Create new revision** > Container `Main` > **Network settings** > **Port mappings** > changed to `8000`, protocol TCP, app protocol HTTP > re-confirmed all environment variables were still present on this same revision > saved (created revision `:6`)
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Update service** > explicitly selected revision `:6` > checked **Force new deployment** > Update

**Result:** Express Services regenerated a fresh target group (`ecs-gateway-tg-7091f4f056919025e`) automatically matching the corrected port (`HTTP : 8000`), the task registered, health checks passed, and:
```bash
curl -i https://le-fc9f3fb04e324cf99566a492452b12b0.ecs.us-east-1.on.aws/api/health
```
returned `200 OK` — backend confirmed live.

---

### Issue 9 — Transient 503 that wasn't actually a new bug

**Symptom:** a 503 briefly reappeared right after triggering the Issue 8 fix.

**Navigation path:**
* Top-of-page notification banner, visible on any ECS Console page while a deployment is active

**What we found there:** the banner read **"Deployment in progress: Express service leadpro-backend-d321 is currently deploying."** The new task was still starting up and hadn't passed its first health check yet — normal, temporary state during a rollout, not a new failure.

**Fix:** none needed — waited roughly 1–2 minutes for the deployment banner to clear, then retried the health check successfully.

---

### Issue 10 — CORS blocked because env vars had a path/trailing slash, not just the origin

**Symptom:** once the frontend was pointed at the ECS backend, login calls failed in the browser console with:
```
Access to fetch at 'https://le-fc9f3fb04e324cf99566a492452b12b0.ecs.us-east-1.on.aws/api/auth/login'
from origin 'https://lead-management-system-two-sigma.vercel.app' has been blocked by CORS policy:
Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin'
header is present on the requested resource.
```

**Navigation path:**
* ECS > Task definitions > `default-leadpro-backend-d321` > latest revision > Container `Main` > **Environment and secrets** — checked the actual values of `CORS_ORIGINS` and `FRONTEND_URL`

**What we found there:** `FRONTEND_URL` had been set with a path appended (`.../login`), and `CORS_ORIGINS` had a trailing slash (`.../`). Neither matches what a browser actually sends — the `Origin` header is always just scheme + host (+ port), never a path or trailing slash — so FastAPI's `CORSMiddleware` never matched the incoming origin against the allow-list and omitted `Access-Control-Allow-Origin` entirely, which the browser then blocks on the preflight.

**Fix — where we navigated:**
* ECS > Task definitions > `default-leadpro-backend-d321` > **Create new revision** > Container `Main` > **Environment and secrets** > corrected both values to the bare origin only:
  ```
  CORS_ORIGINS=https://lead-management-system-two-sigma.vercel.app
  FRONTEND_URL=https://lead-management-system-two-sigma.vercel.app
  ```
* ECS > Clusters > `default` > Express services > `leadpro-backend-d321` > **Update service** > selected the new revision > checked **Force new deployment** > Update

**Lesson:** CORS origin values must be exact scheme+host(+port) matches with no path and no trailing slash — a common copy-paste mistake when grabbing a URL straight from the browser address bar (which often includes the current page path).

---

## Frontend → Vercel (next step)

Frontend stays on Vercel (not ECS) — only the backend moved to AWS. To connect them:

1. Vercel Dashboard → project → **Settings → Environment Variables**
2. Set `NEXT_PUBLIC_API_URL` to:
   ```
   https://le-fc9f3fb04e324cf99566a492452b12b0.ecs.us-east-1.on.aws
   ```
3. Redeploy the frontend (Vercel → Deployments → ⋯ → Redeploy, or push a commit) —
   `NEXT_PUBLIC_*` vars are baked in at **build time**, so just changing the env var
   without redeploying won't take effect.
4. Once redeployed, confirm on the backend's env vars (ECS task definition) that
   `CORS_ORIGINS` and `FRONTEND_URL` match the Vercel production URL
   (`https://lead-management-system-two-sigma.vercel.app`) — otherwise the frontend
   will hit CORS errors calling the new backend.
5. If Google Calendar should work from this deployment, confirm `GOOGLE_REDIRECT_URI`
   on the backend is set to the ECS URL's `/api/calendar/callback` and that exact URI
   is registered in Google Cloud Console → Credentials → Authorized redirect URIs.

## LeadPro Deployment Log — Step by Step (what we actually did)

A chronological record of the concrete steps/commands used to get the LeadPro
backend running on ECS, for repeatability (e.g. redoing this for a new environment,
or replicating for the frontend service).

### Step 1 — Confirm AWS CLI access

```bash
aws sts get-caller-identity --query Account --output text --profile saim-user
```

Confirmed account ID `435472314740`, region `us-east-1`. When this later failed with
`Error when retrieving token from sso: Token has expired and refresh failed`, fixed
with:

```bash
aws sso login --profile saim-user
```

### Step 2 — Create the ECR repository

```bash
aws ecr create-repository --repository-name leadpro-backend --region us-east-1 --profile saim-user
```

(Frontend repo `leadpro-frontend` created the same way, later.)

### Step 3 — Authenticate Docker against ECR

```bash
aws ecr get-login-password --region us-east-1 --profile saim-user | docker login --username AWS --password-stdin 435472314740.dkr.ecr.us-east-1.amazonaws.com
```

### Step 4 — Build and push the backend image (first attempt — had a bug)

```bash
cd backend
docker build -t leadpro-backend .   # ← missing --platform, built arm64 on Apple Silicon
docker tag leadpro-backend:latest 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```

This image later caused `exec /usr/bin/sh: exec format error` on Fargate (amd64
runtime, arm64 image) — see fix section below. Corrected build command used from
then on:

```bash
docker build --platform linux/amd64 -t leadpro-backend .
docker tag leadpro-backend:latest 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```

### Step 5 — Create the service via ECS "Express Services"

Console: ECS → Clusters → `default` → Express services → Create service.
Express Services auto-provisions in one flow: the cluster association, task
definition (`default-leadpro-backend-d321:1`), ALB (`ecs-express-gateway-alb-*`),
listener, target group(s), certificate, and security groups — instead of creating
each piece manually via the "Console checklist" further below (that checklist is
the manual/from-scratch equivalent if Express Services isn't used).

Hit and fixed along the way: `ecsInfrastructureRoleForExpressServices` missing
`ec2:DescribeAccountAttributes`, which stalled ALB/cert/security-group/listener/
target-group provisioning — see fix section below.

### Step 6 — First deployment attempt failed (circuit breaker rollback)

Deployment ID `pAqFD3ipH3ItQ5wt3lJOy` rolled back: "Service deployment rolled back
because the circuit breaker threshold was exceeded." Rollback itself then failed
("No rollback candidate was found") since this was the first-ever deployment — no
earlier healthy revision existed to fall back to.

Root causes found (both applied to task definition revision 1):
1. Image was arm64, not amd64 (Step 4's original build) → zero container logs at all
2. `DATABASE_URL` and other required env vars were never set on the task definition
   at all → app would have crashed on startup even with the right architecture

### Step 7 — Add environment variables via a new task definition revision

Console: Task definitions → `default-leadpro-backend-d321` → select latest
revision → **Create new revision** → container `Main` → Environment variables →
added:

| Key | Value |
|---|---|
| `DATABASE_URL` | Neon pooled connection string |
| `JWT_SECRET` | long random string |
| `ADMIN_EMAIL` | admin login email |
| `ADMIN_PASSWORD` | admin login password |
| `CORS_ORIGINS` | frontend URL(s) |
| `FRONTEND_URL` | frontend URL |

Result: revision `default-leadpro-backend-d321:4` created. (A `:5` was also created
during iteration — always confirm which revision number actually has the complete,
correct env var set before pointing the service at it; don't assume the highest
number is automatically correct.)

### Step 8 — Rebuild and repush the image with the correct architecture

```bash
cd backend
docker build --platform linux/amd64 -t leadpro-backend .
docker tag leadpro-backend:latest 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```

Pushed under the `latest` tag (not a bare digest), so future "Force new deployment"
actions pick it up automatically.

### Step 9 — Point the task definition at the image by tag, not digest

When editing the task definition's container image, used "Select Amazon ECR image"
→ repository `leadpro-backend` → selected the row tagged `latest` (not the
untagged/older digest row) → "Select image by" set to **Image tag** → `latest`.

### Step 10 — Update the service to the corrected revision + force redeploy

Console: Service `leadpro-backend-d321` → **Update service** → Task definition
revision set to `4` (the one with all env vars) → checked **Force new deployment**
→ Update.

### Step 11 — Verify

Check ECS → Service → **Configuration** tab confirms the active Task Definition
field shows the expected revision number. Check the **Logs** tab (time range
widened beyond the default 1 hour) for actual uvicorn/FastAPI startup output rather
than silence. Once a task reaches `RUNNING` and the target group shows it healthy,
hit the public ingress URL:

```bash
curl -i https://<public-ingress-path>/api/health
```

*(In progress as of this doc's last update — confirm this returns `{"status":"ok"}`
before moving on to deploying the frontend service the same way.)*

## ECS Concepts & Guidance

Reference for the core building blocks involved in this deployment, why each one
exists, and what goes wrong when it's misconfigured (with links to the real issues
hit below where relevant).

### Cluster

A logical grouping of compute capacity that your tasks/services run in. With
Fargate, a cluster doesn't "hold" servers you manage — it's just a namespace tying
your services, tasks, and their metrics/logs together. Ours: `default`.

**Why it matters:** everything else (services, tasks, scaling policies) is scoped to
a cluster. Wrong cluster selected = looking in the wrong place for logs/tasks, which
is why "no tasks listed" was confusing earlier — always confirm cluster name first
when something looks missing.

### Task Definition

A versioned blueprint (JSON) describing: which container image to run, CPU/memory,
port mappings, environment variables, IAM roles, and logging config. **Task
definitions are immutable** — you never edit one in place, you create a new
**revision** (e.g. `default-leadpro-backend-d321:4` → `:5`) and point the service at
the new revision.

**Why it matters:** this immutability is exactly why we ended up with revisions 4
and 5 — every env var change (like adding `DATABASE_URL`) requires a fresh revision.
Always double check *which* revision a service is actually running (Service →
Configuration tab) rather than assuming "I edited it" took effect — you can't edit
an existing revision, only supersede it.

### Task

A running instance of a task definition — the actual container(s) executing on
Fargate compute. A task can be `PROVISIONING → PENDING → RUNNING → STOPPED`.

**Why it matters:** when a task fails immediately (bad image architecture, missing
env var, crash on boot), it cycles through and stops so fast it may not even appear
in the console's default filters — which is why we had to widen the time range and
check "Any status" to find stopped tasks and their stop reasons.

### Service

Keeps a desired number of tasks running continuously, replacing any that stop
(crash, get killed, fail health checks), and manages rolling deployments when you
push a new task definition revision. Ours: `leadpro-backend-d321`.

**Why it matters:** a service retrying and failing repeatedly is what triggers the
**deployment circuit breaker** — ECS gives up and rolls back if new tasks can't
reach a healthy state within its bake-time window. That's the "Service deployment
rolled back because the circuit breaker threshold was exceeded" error we hit; the
real fix is always to find *why* the task itself failed (architecture mismatch,
missing `DATABASE_URL`), not to fight the circuit breaker directly.

### Container Image / Amazon ECR

ECR is AWS's private Docker registry. Your task definition references an image by
`<account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`. Images can be selected
by **tag** (mutable, e.g. `latest` — always resolves to whatever was last pushed
with that tag) or by **digest** (immutable, pinned to one exact build).

**Why it matters:** tag vs. digest is a real tradeoff. Digest pinning guarantees
reproducibility but means every new push requires manually reselecting the digest
in the task definition. Tagging as `latest` and selecting "Image tag" mode means
"Force new deployment" always picks up your newest push automatically — this is
what we switched to.

**Critical gotcha we hit:** Docker builds default to your host machine's CPU
architecture. Building on Apple Silicon (arm64) and pushing straight to ECR/Fargate
(which runs amd64 by default) produces an image that fails instantly with
`exec /usr/bin/sh: exec format error` — and because it fails before your app code
even runs, **no logs are written at all**. Zero logs after a deploy is a strong
signal to check architecture first. Always build with `--platform linux/amd64`
explicitly for Fargate.

### IAM Roles (Execution Role vs. Task Role)

Two distinct roles, easy to confuse:
- **Task execution role** (`ecsTaskExecutionRole`) — used by the *ECS agent itself*
  to pull the image from ECR and write logs to CloudWatch, before your app even
  starts. Needs `AmazonECSTaskExecutionRolePolicy`.
- **Task role** — used by *your application code* at runtime to call other AWS
  services (S3, Secrets Manager, etc.), if it needs to. Not required for LeadPro
  currently since Neon/Google are reached over plain internet, not AWS APIs.
- **Infrastructure role** (`ecsInfrastructureRoleForExpressServices`) — specific to
  the "Express Services" console flow; used by ECS itself to auto-provision the
  ALB, certificate, security groups, and target groups on your behalf.

**Why it matters:** each role needs *exactly* the permissions for its job — too few
and provisioning/pulling silently fails (as with the missing
`ec2:DescribeAccountAttributes` permission that stalled ALB/cert/security-group
provisioning), too many is an unnecessary security risk.

### Networking — VPC, Subnets, Security Groups

Fargate tasks run inside subnets of a VPC and get their inbound/outbound traffic
gated by **security groups** (stateful firewalls attached to the task's network
interface, not the instance).

**Why it matters:** the backend's security group should only allow inbound traffic
on its container port (8000) from the *load balancer's* security group — not from
the public internet directly. The ALB's security group is what's open to
0.0.0.0/0 on 80/443. This two-layer setup means the container is never directly
internet-reachable, only through the ALB.

### Load Balancer — ALB, Listener, Target Group

The **Application Load Balancer (ALB)** is the public entry point. A **listener**
watches a port (80/443) and routes matching requests via **listener rules** to a
**target group**, which tracks the health of the actual running tasks and only
routes traffic to ones passing the **health check** (ours: `GET /api/health`).

**Why it matters:** if the health check path/port doesn't match what the container
actually serves, the ALB will never mark a task healthy even if the app is running
fine — targets stay "unhealthy" forever and requests 503. Since our backend listens
on 8000 and serves `/api/health`, both need to match exactly in the target group
config.

### Environment Variables & Secrets

Env vars are set on the **task definition**, injected into the container at
**runtime** by ECS — never baked into the image. This is different from frontend
build-time vars (`NEXT_PUBLIC_*` in Next.js), which get compiled directly into the
JS bundle at `docker build` time and require a rebuild+repush to change.

**Why it matters:** this is exactly the bug we hit — `DATABASE_URL` was never added
to the task definition, so the app crashed on startup trying to connect to
Postgres, before it could serve anything (including the health check). For
anything sensitive (DB passwords, JWT secrets, API keys), the more secure pattern
is referencing **AWS Secrets Manager** or **SSM Parameter Store** from the task
definition instead of plaintext env vars — worth migrating to once the app is
stable, since plaintext env vars are visible to anyone with ECS read access.

### Logging — CloudWatch Logs

Each container's stdout/stderr is shipped to a CloudWatch Logs group (one log
stream per task). This is the single most useful debugging tool for ECS — always
check here before anything else, but remember:

**Why it matters:** if a container fails before it can even start logging (bad
image, execution role can't pull, architecture mismatch), you'll see **zero logs**,
not an error message. Zero logs = something failed before your app code ran.
A crash *inside* your app (bad DB URL format, unhandled exception) *will* show a
Python traceback here — that distinction is the fastest way to narrow down which
category a failure falls into.

### Deployments & the Circuit Breaker

When you update a service (new task def revision, or "Force new deployment"), ECS
starts new tasks alongside old ones and gradually shifts traffic (our setup uses a
**canary** strategy: 5% traffic, 3-minute bake time). If new tasks don't reach a
healthy steady state in time, the **deployment circuit breaker** automatically
stops the rollout and attempts a **rollback** to the previous stable revision.

**Why it matters:** a rollback can itself fail ("No rollback candidate was found")
if there was no previously-healthy revision to fall back to — which is normal on a
service's very first deployment. Don't read "rollback failed" as a second bug;
it just means fix the new revision and redeploy, there's nothing to roll back to.

### Revisions & Redeployment Discipline

Because task definitions are immutable, the practical workflow is always:
edit → **new revision** → update service to point at it → **Force new deployment**.
"Force new deployment" is also needed even without a new revision if you only
changed which image tag/digest is pulled (e.g. pushed a new `:latest`), since ECS
won't otherwise notice the underlying image changed.

**Why it matters:** it's easy to think you "updated" something by editing a form
and saving, but if you didn't select the new revision on the *service* and didn't
force a new deployment, the old (broken) revision keeps running untouched. Always
verify post-change: Service → Configuration tab → confirm the Task Definition field
shows the revision number you expect.

## 1. ECR login

```bash
aws sts get-caller-identity --query Account --output text --profile saim-user

aws ecr get-login-password --region us-east-1 --profile saim-user | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

## 2. ECR repos (already created via console: leadpro-backend, leadpro-frontend pending)

```bash
aws ecr create-repository --repository-name leadpro-backend --region us-east-1 --profile saim-user
aws ecr create-repository --repository-name leadpro-frontend --region us-east-1 --profile saim-user
```

## 3. Backend image build & push

```bash
cd backend
docker build --platform linux/amd64 -t leadpro-backend .
docker tag leadpro-backend:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```

## 4. Frontend image build & push (do this AFTER backend has a public URL)

```bash
cd frontend
docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL=https://<backend-alb-url> \
  -t leadpro-frontend .
docker tag leadpro-frontend:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/leadpro-frontend:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/leadpro-frontend:latest
```

Note: `NEXT_PUBLIC_API_URL` is baked into the JS bundle at build time — changing the
backend URL later requires rebuilding and re-pushing this image, not just an env var change.

## 5. Cluster (if not using console)

```bash
aws ecs create-cluster --cluster-name leadpro-cluster --region us-east-1 --profile saim-user
```

## 6. Networking lookups (useful even when using the console)

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --profile saim-user)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[].SubnetId' --output text --profile saim-user)
echo $VPC_ID
echo $SUBNETS
```

## 7. Task execution IAM role (console: IAM > Roles > Create role > ECS Task, or CLI below)

```bash
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  --profile saim-user

aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
  --profile saim-user
```

## Fix: ECS Express Services provisioning stuck (AccessDenied on ec2:DescribeAccountAttributes)

When deploying via the ECS console's "Express Services" flow, the auto-created role
`ecsInfrastructureRoleForExpressServices` may lack `ec2:DescribeAccountAttributes`,
blocking ALB/cert/security-group/listener/target-group provisioning (all stuck in
"Provisioning" with an AccessDenied message).

```bash
aws iam put-role-policy --role-name ecsInfrastructureRoleForExpressServices \
  --policy-name AllowDescribeAccountAttributes \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"ec2:DescribeAccountAttributes","Resource":"*"}]}' \
  --profile saim-user
```

Provisioning auto-retries after the policy is attached, no redeploy needed.

## Fix: "Cannot perform an interactive login from a non TTY device" / SSO token expired

Docker login started falling back to an interactive username/password prompt, which
fails in a non-interactive shell. Root cause turned out to be an expired AWS SSO
session — `aws ecr get-login-password` was silently failing to return a token, so
`docker login --password-stdin` received an empty password and Docker fell back to
interactive login.

Real error once surfaced: `Error when retrieving token from sso: Token has expired
and refresh failed`.

Fix — re-authenticate the SSO session, then retry the ECR login:

```bash
aws sso login --profile saim-user

aws ecr get-login-password --region us-east-1 --profile saim-user | docker login --username AWS --password-stdin 435472314740.dkr.ecr.us-east-1.amazonaws.com
```

Takeaway: if `docker login` to ECR ever throws a TTY/interactive-login error, check
the AWS SSO session first (`aws sts get-caller-identity --profile saim-user`) before
assuming it's a Docker problem — an expired/empty credential upstream is a common
cause of that exact Docker error message.

## Fix: task definition was missing DATABASE_URL entirely

`DATABASE_URL` is a runtime env var, not baked into the image (unlike the frontend's
NEXT_PUBLIC_API_URL, which IS build-time). It was never set in the task definition,
so the app crashed connecting to Postgres during FastAPI's startup lifespan before
serving anything, including /api/health.

Fix: Task definitions > default-leadpro-backend-d321 > Create new revision > add
env vars on the container: DATABASE_URL (Neon pooled connection string), JWT_SECRET,
ADMIN_EMAIL, ADMIN_PASSWORD, CORS_ORIGINS, FRONTEND_URL (and Google Calendar vars if
needed, updating GOOGLE_REDIRECT_URI to this backend's new domain).

Created revision: `default-leadpro-backend-d321:4`

Then: Service > Update service > select revision 4 > Force new deployment.

## Fix: "exec /usr/bin/sh: exec format error" — architecture mismatch

Container built on Apple Silicon (arm64) but Fargate runs linux/amd64 by default.
Log showed zero output because the container couldn't even exec its entrypoint.

Fix: always build with `--platform linux/amd64` explicitly, then push and force a new
ECS deployment (Update service > Force new deployment).

```bash
docker build --platform linux/amd64 -t leadpro-backend .
docker tag leadpro-backend:latest 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
docker push 435472314740.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest
```

## Console checklist (backend)

1. ECS > Clusters > Create cluster > "leadpro-cluster" (Fargate)
2. ECS > Task definitions > Create new task definition
   - Launch type: Fargate
   - Task role / execution role: ecsTaskExecutionRole
   - Container image: `<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/leadpro-backend:latest`
   - Container port: 8000
   - Env vars: DATABASE_URL, JWT_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD, CORS_ORIGINS,
     GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI, FRONTEND_URL
     (use Secrets Manager / SSM references for anything sensitive rather than plaintext)
3. EC2 > Load Balancers > Create Application Load Balancer
   - Target group: type IP, port 8000, health check path `/api/health`
4. ECS > Clusters > leadpro-cluster > Create service
   - Task definition: the one above
   - Desired tasks: 1 (scale later)
   - Attach to the ALB target group created above
   - Security group: allow inbound 8000 from ALB's security group only; ALB SG allows
     inbound 80/443 from 0.0.0.0/0
5. Once RUNNING, get the ALB DNS name (EC2 > Load Balancers) and test:
   `curl http://<alb-dns-name>/api/health`

---

## Health Hub — ECS Deployment Log

Second project deployed to the same ECS cluster (`default`), same account. Stack:
Django (gunicorn) web app + Celery worker + Redis broker, Postgres already on Neon
(same pattern as LeadPro). Image was already built and pushed to its own ECR repo
(`health-hub`) before this log starts — dockerization itself is out of scope here,
this covers ECS setup and getting the web service to a healthy state.

**Status as of last update: web service still being fixed — `/health` endpoint was
missing entirely (HH-6), added in code. Also fixed: migrations never ran in prod
because they only lived in docker-compose's command override (HH-7), now baked
into the image via `entrypoint.sh`. Image rebuild/redeploy in progress.**
Redis and Celery worker services not yet created (planned: Redis via
`public.ecr.aws/docker/library/redis:7`, worker via the same `health-hub` image
with a `celery -A config worker -l info` command override, both connected to the
web service via ECS Service Connect — no ALB/target group needed for either).

### HH-1 — Health check grace period was 0 seconds

**Symptom:** gunicorn logs showed clean, successful boots every time —
`Starting gunicorn`, `Listening at: http://0.0.0.0:8000`, `Booting worker` — then a
few minutes later `Handling signal: term` / `Worker exiting` / `Shutting down:
Master`, repeating in a loop with a new task ID each cycle. No Python traceback
anywhere — the app itself never crashed.

**Navigation path:**
* ECS > Clusters > `default` > Express services > `health-hub-1a26` > **Health and metrics** tab (Express service overview page) — showed **"Health check grace period: 0 seconds"**

**What we found there:** with a 0-second grace period, ECS starts judging task
health immediately on start, before gunicorn/Django has finished initializing. The
ALB's first health check attempt(s) can hit the container before it's actually
ready, and with no grace period that immediately counts as a failure — ECS then
kills and replaces the task before it ever gets a real chance to stabilize.

**Fix:**
* ECS > Clusters > `default` > Express services > `health-hub-1a26` > **Update service** > set **Health check grace period** to `60` seconds > Force new deployment

**Result:** ruled out timing as the (sole) cause — the crash loop continued after
this fix, confirming the app was also failing its health checks once genuinely up,
not just failing due to being checked too early. Kept the 60s grace period anyway
since it's correct regardless.

### HH-2 — Target group health check failing with `[400]`

**Symptom:** crash loop continued even after the grace period fix.

**Navigation path:**
* EC2 > Load Balancing > Target Groups > the `health-hub` target group > **Targets** tab > **Health status details** column — showed `Health checks failed with these codes: [400]`

**What we found there:** a `400` from a Django app almost always means
`DisallowedHost` — the ALB sends its own DNS name as the `Host` header on health
check requests, not the app's real domain, and if that hostname isn't in Django's
`ALLOWED_HOSTS`, every request (including health checks) gets rejected with `400`
before routing even happens.

**Fix:**
* ECS > Task definitions > `default-health-hub-1a26` > **Create new revision** > Environment variables > added `DJANGO_ALLOWED_HOSTS=*` (matching the exact env var name the app's `prod.py` reads via `env("DJANGO_ALLOWED_HOSTS")` — note this is *not* named plain `ALLOWED_HOSTS`, an early wrong guess that would have done nothing)

### HH-3 — `DJANGO_SETTINGS_MODULE` was never set on the task definition

**Symptom:** uncertainty over which Django settings file was actually active in
production — `dev.py` (`ALLOWED_HOSTS = ["*"]`, insecure defaults, matches
docker-compose's local setup) vs `prod.py` (secure hardened settings, requires
`DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS` from env with no fallback).

**Navigation path:**
* ECS > Task definitions > `default-health-hub-1a26` (current revision) > Environment and secrets — `DJANGO_SETTINGS_MODULE` was absent entirely

**What we found there:** relying on whatever implicit default is baked into
`manage.py`/`wsgi.py` for something this consequential (security headers, `DEBUG`,
`ALLOWED_HOSTS`, database config) is exactly the kind of ambiguity that produces
inconsistent, hard-to-diagnose behavior between deploys.

**Fix:**
* ECS > Task definitions > `default-health-hub-1a26` > **Create new revision** > Environment variables > added `DJANGO_SETTINGS_MODULE=config.settings.prod` explicitly, alongside confirming `DJANGO_SECRET_KEY` was present

### HH-4 — Couldn't select/deploy a specific revision in "Image digest" mode

**Symptom:** while updating the service to point at the new task definition
revision, the "Select Amazon ECR image" dialog wouldn't let anything be picked
when **Select image by** was set to **Image digest**.

**Navigation path:**
* Same Update service flow > "Select Amazon ECR image" dialog > **Select image by** section

**Fix:**
* Switched **Select image by** from **Image digest** to **Image tag** > selected the `latest` tag (only tag that existed, since only one image had been pushed) > continued through the rest of the update flow normally

**Lesson:** same as LeadPro Issue 9's underlying setup — Image tag mode is both
more reliable in this picker and means future `latest` pushes get picked up
automatically on a Force new deployment, without needing to reselect anything.

### HH-5 — Target group health check failing with `[301]`

**Symptom:** after fixing HH-2/HH-3 and redeploying, the crash loop stopped but the
public URL returned `503 Service Temporarily Unavailable`.

**Navigation path:**
* EC2 > Load Balancing > Target Groups > the `health-hub` target group > Targets tab > **Health status details** — showed `Health checks failed with these codes: [301]`
* Also checked (to rule out): task definition's container port (`8000`, correct) and target group's own configured port (`8000`, correct) — confirmed this was not a port mismatch like LeadPro's Issue 8

**What we found there:** `prod.py` has `SECURE_SSL_REDIRECT = env.bool("SECURE_SSL_REDIRECT", default=True)`. ALB health checks hit the container directly over plain HTTP, without the `X-Forwarded-Proto: https` header real browser traffic gets — so Django saw a non-HTTPS request and issued a `301` redirect to HTTPS, which the ALB's health check does not follow, reading as a failure regardless of the app being otherwise fine.

**Fix:**
* ECS > Task definitions > `default-health-hub-1a26` > **Create new revision** > Environment variables > added `SECURE_SSL_REDIRECT=False` (already env-configurable, no code change needed for this one)

**Why this is an acceptable fix here, not just a workaround:** the ALB itself
terminates public HTTPS traffic (Express Services provisioned an ACM certificate on
the listener) — the internal hop from ALB to container over plain HTTP happens
inside the VPC, so Django's own SSL-redirect enforcement is largely redundant in
this architecture. A more surgical alternative for later: keep
`SECURE_SSL_REDIRECT=True` and add `SECURE_REDIRECT_EXEMPT = [r"^health/?$"]` in
`prod.py` to exempt only the health check path — that requires a code change and
image rebuild, so it was deferred in favor of the immediate env-var fix.

### HH-6 — Target group health check failing with `[404]`

**Symptom:** after the HH-5 fix, `301` was gone but health checks now failed with
`404` — a genuine "not found," not a redirect or rejection this time.

**Navigation path:**
* Same Target Groups > Targets tab > Health status details — `Health checks failed with these codes: [404]`

**What we found there:** with the SSL-redirect middleware no longer intercepting
the request first, it reached Django's URL router — which had no route registered
at `/health` at all. Unlike HH-1 through HH-5, this one is a genuine code gap, not
a config/env var issue, and requires an actual code change + image rebuild.

**Fix (in progress):**
* Added `apps/views.py`:
  ```python
  from django.http import HttpResponse

  def health_check(request):
      return HttpResponse("OK", status=200)
  ```
* Wired into the project's `urls.py`:
  ```python
  from apps.views import health_check

  urlpatterns = [
      path("health", health_check),  # no trailing slash — matches target group's configured path exactly
      # ...existing urlpatterns
  ]
  ```
* Next: rebuild image with `--platform linux/amd64`, push to the `health-hub` ECR repo under the `latest` tag, then Update service > Force new deployment, and recheck target group health status.

**Lesson carried over from LeadPro:** each of HH-2, HH-5, and HH-6 produced a
*different* HTTP status code from the target group (`400` → `301` → `404`) as each
underlying issue got fixed in turn — the status code in "Health status details" is
the single most direct signal for what's actually wrong at each step; always check
it fresh after every fix rather than assuming the same root cause persists.

### HH-7 — Migrations never ran in production

**Symptom:** app was healthy (`/health` returning `200`) but database tables from
new migrations were missing/stale in prod — `Seems like migrations didnt run on prod`.

**Navigation path:**
* No AWS console navigation this time — this was a Dockerfile/codebase issue,
  found by comparing `docker-compose.yml`'s `web` service `command:` against the
  Dockerfile's own `CMD`/`ENTRYPOINT`.

**What we found there:** `docker-compose.yml` ran migrations via a `command:`
override (`sh -c "python manage.py migrate && gunicorn ..."`) that only applies
when the container is started **through that specific compose file**. The
Dockerfile's own baked-in startup behavior never included a migrate step, so any
standalone run of the image — including every ECS task — skipped migrations
entirely and went straight to `gunicorn`.

**Fix:** moved the startup logic into the codebase itself via an `entrypoint.sh`,
baked into the image with `ENTRYPOINT`, so migrations run automatically on every
container start regardless of environment (local, ECS, or otherwise) — no manual
ECS console "Command" override needed:

```bash
#!/bin/sh
set -e

python manage.py migrate --noinput

exec gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

The `exec` matters — it replaces the shell process with gunicorn so `SIGTERM`
from ECS during deploys/scale-in reaches gunicorn directly for a graceful
shutdown, instead of being swallowed by an intermediate shell.

**Gotchas hit while wiring this in:**
* A `COPY entrypoint.sh /entrypoint.sh` and a `COPY requirements/ requirements/`
  must be **two separate `COPY` instructions** — `COPY` only takes one
  destination at the end of the line, so merging both copies into one
  instruction via a line-continuation silently copies the wrong things into the
  wrong place.
* `RUN chmod +x /entrypoint.sh && pip install ...` needs the `&&` — without it, a
  line-continuation backslash merges both into one broken shell command where
  `pip`/`install`/etc. get passed to `chmod` as bogus extra filenames, and `pip
  install` never actually runs.
* `RUN python manage.py collectstatic ...` needs `DJANGO_SECRET_KEY` and
  `DJANGO_ALLOWED_HOSTS` to be resolvable at **build time** (Django imports
  `prod.py` to run the management command), but ECS only supplies those at
  **container runtime**. Fixed with build-time-only placeholder `ARG`s that get
  fully overridden by the real ECS task definition env vars once the container
  actually runs:
  ```dockerfile
  ARG DJANGO_SECRET_KEY=build-time-placeholder
  ARG DJANGO_ALLOWED_HOSTS=localhost
  ENV DJANGO_SECRET_KEY=$DJANGO_SECRET_KEY \
      DJANGO_ALLOWED_HOSTS=$DJANGO_ALLOWED_HOSTS
  ```
  Previously this line silently failed and was masked by a trailing `|| true`,
  meaning static files were likely never actually being collected into the image.

**Bonus fix:** baking `DJANGO_SETTINGS_MODULE=config.settings.prod` into the
image's own `ENV` also permanently closes out HH-3 — the ECS task definition no
longer needs a manual override for it, since the image now defaults correctly on
its own.

### HH-8 — `entrypoint.sh` path mismatch between `COPY`, `chmod`, and `ENTRYPOINT`

**Symptom:** after moving `entrypoint.sh` into a `scripts/` folder in the repo,
the Dockerfile was updated inconsistently across the three lines that reference
it, which would fail the build (`chmod`: no such file or directory) or fail the
container at start (`ENTRYPOINT`: exec format/not found).

**Navigation path:** none — caught by reading the Dockerfile directly, no AWS
console involved. `COPY` takes two arguments, `<source> <destination>`; the
destination is whatever you type as the second argument, not automatically the
same as the source path. `chmod`/`ENTRYPOINT` paths with no leading `/` are
*relative to `WORKDIR`* (`/app` in this Dockerfile), not relative to the repo.

**What we found there:** the file went through several edits and ended up with
three different effective paths in the same Dockerfile:

```dockerfile
COPY scripts/entrypoint.sh /entrypoint.sh      # → lands at /entrypoint.sh
RUN chmod +x scripts/entrypoint.sh             # → looks for /app/scripts/entrypoint.sh (doesn't exist)
ENTRYPOINT ["scripts/entrypoint.sh"]           # → looks for /app/scripts/entrypoint.sh (doesn't exist)
```

`COPY`'s destination (`/entrypoint.sh`) doesn't automatically match the source
path (`scripts/entrypoint.sh`) — the two are independent, and the *destination*
is what every later line needs to reference.

**Fix:** use the exact same absolute path (leading `/`) in the `COPY`
destination, the `chmod` target, and `ENTRYPOINT` — source path can differ from
destination, but all three destination references must be identical:

```dockerfile
COPY scripts/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && pip install --no-cache-dir -r requirements/base.txt

ENTRYPOINT ["/entrypoint.sh"]
```

**Lesson:** whenever an entrypoint/startup script is referenced in more than one
Dockerfile instruction, treat the destination path as a single source of truth —
copy it once, then reuse that literal string everywhere else, ideally as an
absolute path so `WORKDIR` changes can't silently break it.

### Planned next steps for Health Hub

1. Confirm `/health` returns `200` and the target group shows healthy.
2. Create the Redis service: image `public.ecr.aws/docker/library/redis:7` (typed
   directly into the image field, not browsed from ECR since it's not a private
   repo), port `6379`, no load balancer/target group, ECS Service Connect enabled
   with service name `redis`.
3. Create the Celery worker service: same `health-hub` ECR image as web, container
   Command overridden to `celery -A config worker -l info`, same env vars as web
   (`DATABASE_URL`, `DJANGO_SETTINGS_MODULE`, `DJANGO_ALLOWED_HOSTS`,
   `CELERY_BROKER_URL=redis://redis:6379/0`,
   `CELERY_RESULT_BACKEND=redis://redis:6379/1`), no load balancer/target group,
   Service Connect enabled as a client so it can resolve `redis:6379`.
4. Enable Service Connect on the web service too (as a client) so it can reach
   Redis the same way, if the web app itself also talks to Redis directly (e.g.
   caching), not just via Celery.
5. Security group: Redis's security group needs an inbound rule allowing port
   6379 from the web and worker services' security groups only — never from
   `0.0.0.0/0`.
