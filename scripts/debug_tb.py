import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    base_dir = os.path.expanduser("~/rl_research/auv/master_curriculum")
    print(f"🔎 Scanning for TensorBoard files in: {base_dir}\n")

    search_pattern = os.path.join(base_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(search_pattern, recursive=True)

    if not event_files:
        print("❌ No tfevents files found at all. The directory might be empty.")
        return

    for f in event_files:
        print("-" * 60)
        print(f"📄 File: {f}")
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if tags:
                print(f"✅ Found {len(tags)} scalar tags. First 5 tags:")
                for tag in tags[:5]:
                    print(f"   - {tag}")

                # Specifically check for our targets
                has_sr = "rollout/success_rate" in tags
                has_curr = "cdr/curriculum_level" in tags
                print(f"🎯 Has rollout/success_rate? {has_sr}")
                print(f"🎯 Has cdr/curriculum_level? {has_curr}")
            else:
                print("⚠️ File is empty or has no scalar tags.")
        except Exception as e:
            print(f"❌ Error reading file: {e}")


if __name__ == "__main__":
    main()
