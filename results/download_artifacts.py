# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import requests
import argparse
import datetime
import subprocess
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_token(cli_token):
    # Priority 1: Command-line argument
    if cli_token:
        return cli_token

    # Priority 2: Check for a token file in the home directory (~/.ghtoken)
    token_file = os.path.expanduser("~/.ghtoken")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                token = f.read().strip()
                if token:
                    return token
        except Exception as e:
            print("Error reading token from ~/.ghtoken:", e)

    # Priority 3: Environment variable
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        return env_token

    # Priority 4: Fallback to GitHub CLI (gh)
    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if token:
            return token
    except Exception as e:
        print("Could not retrieve token using GitHub CLI:", e)

    return None


def download_artifact(artifact, folder_name, headers, args, session):
    """
    Download the artifact ZIP file and return its file path.
    Unzipping is deferred.
    """
    artifact_name = artifact["name"]
    if artifact_name in ["install-artifacts"]:
        return None
    if args.filter and args.filter not in artifact_name:
        return None

    artifact_id = artifact["id"]
    # Destination ZIP file: use artifact_name + ".zip"
    target_zip = os.path.join(folder_name, f"{artifact_name}.zip")
    print(f"Downloading artifact '{artifact_name}' (ID: {artifact_id}) to {target_zip}")
    download_url = (
        f"https://api.github.com/repos/{args.repo}/actions/artifacts/{artifact_id}/zip"
    )
    try:
        with session.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(target_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return target_zip
    except Exception as e:
        print(f"Failed to download artifact '{artifact_name}': {e}")
        return None


def process_zip_file(zip_path, folder_name):
    """
    Unzip the given ZIP file serially. For ZIP files with names like
    "full-logs-<model_name>.zip", if the extracted contents include
    "full_job_output.log", rename it to "<model_name>_full_job_output.log".
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder_name)
        os.remove(zip_path)

        zip_base = os.path.basename(zip_path)
        # Check for the special pattern "full-logs-<model_name>.zip"
        if zip_base.startswith("full-logs-") and zip_base.endswith(".zip"):
            # Extract <model_name> from the zip filename.
            # E.g., "full-logs-vision-misc.zip" gives model_name = "vision-misc"
            model_name = zip_base[len("full-logs-") : -len(".zip")]
            extracted_log = os.path.join(folder_name, "full_job_output.log")
            if os.path.exists(extracted_log):
                new_name = os.path.join(
                    folder_name, f"{model_name}_full_job_output.log"
                )
                os.rename(extracted_log, new_name)
                print(f"Renamed {extracted_log} to {new_name}")
    except Exception as e:
        print(f"Failed to process zip file '{zip_path}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions artifacts for the latest run on a given branch."
    )
    parser.add_argument(
        "--repo",
        default="tenstorrent/tt-torch",
        help="Repository in owner/repo format (default: tenstorrent/tt-torch)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to filter workflow runs (default: main)",
    )
    parser.add_argument(
        "--workflow",
        default="nightly-tests.yml",
        help="Optional: Specify the workflow file name to filter runs (default: nightly-tests.yml)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Optional: Filter artifacts by a substring in their name",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub Personal Access Token (if not provided, the script will try to retrieve one using GitHub CLI or environment variable)",
    )
    parser.add_argument(
        "--list", action="store_true", help="Just list artifacts without downloading"
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip downloaded .zip files and remove the original ZIP file. Also apply special renaming if needed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )

    args = parser.parse_args()

    token = get_token(args.token)
    if not token:
        print(
            "Error: GitHub token is required. Provide it with --token, set GITHUB_TOKEN, or login via 'gh auth login'."
        )
        exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Determine the URL to fetch runs.
    runs_url = f"https://api.github.com/repos/{args.repo}/actions/workflows/{args.workflow}/runs"

    # Remove the .yml/.yaml extension from the workflow filename for folder naming.
    workflow_name = os.path.splitext(args.workflow)[0]
    print("Fetching workflow runs for", workflow_name)

    params = {"branch": args.branch, "per_page": 1}  # fetch only the latest run
    runs_response = requests.get(runs_url, headers=headers, params=params)
    if runs_response.status_code != 200:
        print("Failed to get workflow runs:", runs_response.text)
        exit(1)

    runs_data = runs_response.json()
    if not runs_data.get("workflow_runs"):
        print(f"No workflow runs found on branch '{args.branch}'")
        exit(1)

    latest_run = runs_data["workflow_runs"][0]
    run_id = latest_run["id"]
    created_at = latest_run["created_at"]
    date_str = datetime.datetime.fromisoformat(created_at.rstrip("Z")).strftime(
        "%Y%m%d"
    )
    folder_name = f"{workflow_name}_artifacts_{date_str}_run_id_{run_id}"
    os.makedirs(folder_name, exist_ok=True)
    print(
        f"Downloading artifacts for run {run_id} (created on {date_str}) into folder '{folder_name}'."
    )

    # List the artifacts for the specified run using pagination.
    artifacts_url = (
        f"https://api.github.com/repos/{args.repo}/actions/runs/{run_id}/artifacts"
    )
    all_artifacts = []
    page = 1
    while True:
        params = {"per_page": 100, "page": page}
        artifacts_response = requests.get(artifacts_url, headers=headers, params=params)
        if artifacts_response.status_code != 200:
            print("Failed to get artifacts:", artifacts_response.text)
            exit(1)
        artifacts_data = artifacts_response.json()
        artifacts = artifacts_data.get("artifacts", [])
        if not artifacts:
            break
        all_artifacts.extend(artifacts)
        if len(artifacts) < 100:
            break
        page += 1

    if not all_artifacts:
        print("No artifacts found for run", run_id)
        exit(0)

    # If --list is set, list artifact information and exit without downloading.
    if args.list:
        for artifact in all_artifacts:
            artifact_name = artifact["name"]
            if artifact_name in ["install-artifacts"]:
                continue
            if args.filter and args.filter not in artifact_name:
                continue
            artifact_id = artifact["id"]
            download_url = f"https://api.github.com/repos/{args.repo}/actions/artifacts/{artifact_id}/zip"
            print(
                f"Found artifact '{artifact_name}' (ID: {artifact_id}) download_url: {download_url}"
            )
        exit(0)

    downloaded_zips = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = []
            for artifact in all_artifacts:
                futures.append(
                    executor.submit(
                        download_artifact, artifact, folder_name, headers, args, session
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                if result:  # result is the file path for the downloaded ZIP
                    downloaded_zips.append(result)

    # If --unzip is set, process each ZIP file one-at-a-time.
    if args.unzip:
        for zip_file in downloaded_zips:
            process_zip_file(zip_file, folder_name)


if __name__ == "__main__":
    main()
