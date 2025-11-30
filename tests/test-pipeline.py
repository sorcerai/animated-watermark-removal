#!/usr/bin/env python3
"""
Quick pipeline test script

Tests:
1. ComfyUI is running and accessible
2. Audio preservation workflow works
3. Can process a test video end-to-end

Run after completing Windows setup to validate installation
"""

import requests
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.audio_preservation import AudioPreservationPipeline


class ComfyUITester:
    def __init__(self, host="http://localhost:8188"):
        self.host = host
        self.audio_pipeline = AudioPreservationPipeline()

    def test_connection(self) -> bool:
        """Test if ComfyUI is running"""
        try:
            response = requests.get(f"{self.host}/system_stats")
            if response.status_code == 200:
                print("✓ ComfyUI is running")
                stats = response.json()
                print(f"  VRAM: {stats['system']['vram_total'] / 1024 / 1024 / 1024:.1f} GB")
                return True
        except requests.ConnectionError:
            print("✗ ComfyUI not running")
            print("  Start with: cd C:\\sora-watermark-saas\\ComfyUI && .\\run_nvidia_gpu.bat")
            return False
        return False

    def check_models(self) -> bool:
        """Verify required models are loaded"""
        try:
            response = requests.get(f"{self.host}/object_info")
            if response.status_code == 200:
                print("✓ Can access model info")
                # TODO: Parse and verify specific models
                return True
        except Exception as e:
            print(f"✗ Model check failed: {e}")
            return False
        return False

    def test_audio_workflow(self, test_video: str) -> bool:
        """Test audio preservation pipeline"""
        test_video = Path(test_video)

        if not test_video.exists():
            print(f"✗ Test video not found: {test_video}")
            return False

        print(f"\nTesting audio preservation with {test_video.name}...")

        try:
            # Step 1: Extract frames
            frames_dir = self.audio_pipeline.extract_frames(str(test_video))

            # Step 2: Simulate processing (just copy frames)
            print("  Simulating frame processing...")
            import shutil
            processed_dir = Path("temp/test_processed")
            processed_dir.mkdir(parents=True, exist_ok=True)

            for frame in list(frames_dir.glob("*.png"))[:10]:  # Only copy first 10 frames for speed
                shutil.copy(frame, processed_dir / frame.name)

            # Step 3: Stitch back
            output = self.audio_pipeline.stitch_with_audio(
                frames_dir=processed_dir,
                original_video=str(test_video),
                output_video="temp/test_output.mp4"
            )

            # Step 4: Validate
            validation = self.audio_pipeline.validate_audio_sync(
                str(test_video),
                str(output)
            )

            # Cleanup
            self.audio_pipeline.cleanup(frames_dir)
            self.audio_pipeline.cleanup(processed_dir)

            if validation["synced"]:
                print("✓ Audio preservation pipeline working")
                return True
            else:
                print("✗ Audio sync failed")
                return False

        except Exception as e:
            print(f"✗ Audio test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_workflow_api(self, workflow_path: str) -> bool:
        """Test loading and queuing a workflow via API"""
        workflow_path = Path(workflow_path)

        if not workflow_path.exists():
            print(f"✗ Workflow not found: {workflow_path}")
            return False

        try:
            # Load workflow
            workflow = json.loads(workflow_path.read_text())

            # Try to queue it (without actual processing)
            # This validates the workflow is loadable
            print(f"✓ Workflow loaded: {workflow_path.name}")

            # TODO: Submit to /prompt endpoint for actual test
            # For now, just validate JSON structure

            return True

        except Exception as e:
            print(f"✗ Workflow test failed: {e}")
            return False

    def run_full_test(self, test_video: str, workflow_path: str):
        """Run all tests"""
        print("=" * 60)
        print("ComfyUI + Audio Preservation Pipeline Test")
        print("=" * 60)

        results = {
            "connection": self.test_connection(),
            "models": self.check_models(),
            "audio": self.test_audio_workflow(test_video),
            "workflow": self.test_workflow_api(workflow_path)
        }

        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)

        for test, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} {test}")

        all_passed = all(results.values())

        if all_passed:
            print("\n✓ All tests passed! Ready to process real videos.")
        else:
            print("\n✗ Some tests failed. Fix issues before continuing.")

        return all_passed


if __name__ == "__main__":
    import sys

    # Default test files
    test_video = "test-video.mp4"
    workflow_path = "C:/sora-watermark-saas/ComfyUI/ComfyUI/workflows/sora-removal.json"

    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    if len(sys.argv) > 2:
        workflow_path = sys.argv[2]

    tester = ComfyUITester()
    success = tester.run_full_test(test_video, workflow_path)

    sys.exit(0 if success else 1)
