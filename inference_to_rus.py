import argparse
import os

from src.model import VideoDubber


def main():
    video_dubber = VideoDubber()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the source video file",
        required=True,
    )
    parser.add_argument(
        "--source_language",
        type=str,
        help=f"Source language, e.g. english."
        "Now the following languages are supported:"
        f" {list(video_dubber.spacy_models.keys())}",
        default="english",
    )
    args = parser.parse_args()

    audio_file = video_dubber.extract_audio_from_video(args.input)
    if audio_file is None:
        return

    transcription = video_dubber.transcribe_audio(
        audio_file, args.source_language.lower()
    )
    if transcription is None:
        return

    merged_audio, ducked_audio = video_dubber.merge_audio_files(
        transcription,
        args.source_language.lower(),
        args.voice[:5],
        args.voice,
        audio_file,
    )
    if merged_audio is None:
        return
    video_dubber.replace_audio_in_video(args.input, ducked_audio)
    # Save the audio file
    output_filename = os.path.splitext(args.input)[0] + ".wav"
    video_dubber.save_audio_to_file(merged_audio, output_filename)


if __name__ == "__main__":
    main()
