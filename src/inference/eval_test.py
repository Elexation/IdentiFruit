"""
eval_test.py — Evaluates the trained model against data/Test/.
"""

from pathlib import Path
from PIL import Image

from src.inference.predict import predict
from src.training.dataset import IMG_EXTS

TEST_DIR = Path("data/Test")

# Maps each test subfolder name -> the class label the model should predict.
# Known fruits use the model's "fruit_freshness" label format.
# Vegetables the model was never trained on should be rejected as "unknown".
FOLDER_TO_CLASS = {
	# Known fruit classes — model should predict these correctly
	"fresh_apples":    "apple_fresh",
	"rotten_apple":    "apple_rotten",
	"fresh_banana":    "banana_fresh",
	"rotten_banana":   "banana_rotten",
	"fresh_orange":    "orange_fresh",
	"rotten_orange":   "orange_rotten",
	# Out-of-distribution — model should reject these as "unknown"
	"fresh_cucumber":  "unknown",
	"rotten_cucumber": "unknown",
	"fresh_okra":      "unknown",
	"rotten_okra":     "unknown",
	"fresh_potato":    "unknown",
	"rotten_potato":   "unknown",
	"fresh_tomato":    "unknown",
	"rotten_tomato":   "unknown",
	"cars":            "unknown",
	"animals":         "unknown"
}


def prediction_to_class(result: dict) -> str:
	"""
	Converts a predict() result dict into a class label string
	that matches the format used in FOLDER_TO_CLASS.

	predict() returns:
	  {"fruit": "Apple",   "freshness": "fresh",  "confidence": 0.94}
	  {"fruit": "Unknown", "freshness": None,      "confidence": None}

	convert to:
	  "apple_fresh"  (lowercase fruit + _ + freshness)
	  "unknown"
	"""
	if result["fruit"] == "Unknown":
		return "unknown"
	return f"{result['fruit'].lower()}_{result['freshness']}"


def collect_images(folder: Path) -> list[Path]:
	"""Returns all image files in a folder (non-recursive)."""
	return [p for p in sorted(folder.iterdir())
	        if p.is_file() and p.suffix.lower() in IMG_EXTS]


def main():
	# Verify test dir exists
	if not TEST_DIR.exists():
		print(f"ERROR: {TEST_DIR} does not exist.")
		return

	# Collect all test folders that we have a mapping for.
	# Warn about any folders that exist but aren't in the mapping.
	folders = []
	for folder in sorted(TEST_DIR.iterdir()):
		if not folder.is_dir():
			continue
		if folder.name not in FOLDER_TO_CLASS:
			print(f"WARNING: No mapping for folder '{folder.name}' — skipping.")
			continue
		folders.append(folder)

	if not folders:
		print("No recognized test folders found.")
		return

	folder_images = {folder: collect_images(folder) for folder in folders}
	total_images = sum(len(imgs) for imgs in folder_images.values())
	print(f"Evaluating {total_images:,} images across {len(folders)} folders...")
	print()

	# Define column widths for formatted output
	COL_FOLDER   = 18
	COL_EXPECTED = 16
	COL_TOTAL    = 7
	COL_CORRECT  = 8
	COL_ACC      = 9
	COL_CONF     = 8

	header = (
		f"{'Folder':<{COL_FOLDER}}  "
		f"{'Expected':<{COL_EXPECTED}}  "
		f"{'Total':>{COL_TOTAL}}  "
		f"{'Correct':>{COL_CORRECT}}  "
		f"{'Accuracy':>{COL_ACC}}  "
		f"{'Avg Conf':>{COL_CONF}}"
	)
	separator = "─" * len(header)

	print(header)
	print(separator)

	overall_total   = 0
	overall_correct = 0

	known_total   = 0
	known_correct = 0
	unk_total     = 0
	unk_correct   = 0

	processed = 0

	for folder in folders:
		expected_class = FOLDER_TO_CLASS[folder.name]
		images         = folder_images[folder]

		folder_total      = 0
		folder_correct    = 0
		folder_conf       = 0.0
		folder_conf_count = 0

		for img_path in images:
			# Progress counter — 7k images takes a few minutes
			if processed > 0 and processed % 100 == 0:
				print(f"  [{processed:,}/{total_images:,}] processing...", end="\r")

			try:
				with Image.open(img_path) as img:
					result = predict(img)
			except Exception as e:
				print(f"\nWARNING: Skipping {img_path.name} — {e}")
				continue

			pred_class = prediction_to_class(result)
			correct    = (pred_class == expected_class)

			folder_total   += 1
			folder_correct += int(correct)
			if result["confidence"] is not None:
				folder_conf       += result["confidence"]
				folder_conf_count += 1
			processed += 1

		# Accumulate totals
		overall_total   += folder_total
		overall_correct += folder_correct

		if expected_class == "unknown":
			unk_total   += folder_total
			unk_correct += folder_correct
		else:
			known_total   += folder_total
			known_correct += folder_correct

		# Print row
		acc      = folder_correct / folder_total if folder_total else 0
		avg_conf = folder_conf / folder_conf_count if folder_conf_count else 0

		print(
			f"\r{folder.name:<{COL_FOLDER}}  "
			f"{expected_class:<{COL_EXPECTED}}  "
			f"{folder_total:>{COL_TOTAL},}  "
			f"{folder_correct:>{COL_CORRECT},}  "
			f"{acc:>{COL_ACC - 1}.1%}  "
			f"{avg_conf:>{COL_CONF}.3f}"
		)

	# Print overall summary
	overall_acc = overall_correct / overall_total if overall_total else 0
	known_acc   = known_correct   / known_total   if known_total   else 0
	unk_acc     = unk_correct     / unk_total     if unk_total     else 0

	print(separator)
	print(
		f"{'OVERALL':<{COL_FOLDER}}  "
		f"{'':<{COL_EXPECTED}}  "
		f"{overall_total:>{COL_TOTAL},}  "
		f"{overall_correct:>{COL_CORRECT},}  "
		f"{overall_acc:>{COL_ACC - 1}.1%}"
	)
	print()
	print(f"  Known fruit accuracy  ({known_total:,} images):  {known_acc:.1%}")
	print(f"  Unknown accuracy      ({unk_total:,} images):  {unk_acc:.1%}")
	print()


if __name__ == "__main__":
	main()
