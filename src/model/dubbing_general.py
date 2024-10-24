import os
import re
import tempfile
import uuid

import numpy as np
import scipy
import torch

# from google.cloud import texttospeech
# from google.cloud import translate_v2 as translate
import whisper
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel


class VideoDubber:
    def __init__(self):
        self.spacy_models = {
            "english": "en_core_web_sm",
            "german": "de_core_news_sm",
            "french": "fr_core_news_sm",
            "italian": "it_core_news_sm",
            "catalan": "ca_core_news_sm",
            "chinese": "zh_core_web_sm",
            "croatian": "hr_core_news_sm",
            "danish": "da_core_news_sm",
            "dutch": "nl_core_news_sm",
            "finnish": "fi_core_news_sm",
            "greek": "el_core_news_sm",
            "japanese": "ja_core_news_sm",
            "korean": "ko_core_news_sm",
            "lithuanian": "lt_core_news_sm",
            "macedonian": "mk_core_news_sm",
            "polish": "pl_core_news_sm",
            "portuguese": "pt_core_news_sm",
            "romanian": "ro_core_news_sm",
            "russian": "ru_core_news_sm",
            "spanish": "es_core_news_sm",
            "swedish": "sv_core_news_sm",
            "ukrainian": "uk_core_news_sm",
        }

        self.ISWORD = re.compile(r".*\w.*")

        self.ABBREVIATIONS = {
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "No.": "Number",
            "Dr.": "Doctor",
            "Ms.": "Miss",
            "Ave.": "Avenue",
            "Blvd.": "Boulevard",
            "Ln.": "Lane",
            "Rd.": "Road",
            "a.m.": "before noon",
            "p.m.": "after noon",
            "ft.": "feet",
            "hr.": "hour",
            "min.": "minute",
            "sq.": "square",
            "St.": "street",
            "Asst.": "assistant",
            "Corp.": "corporation",
        }

    def extract_audio_from_video(self, video_file):
        try:
            print("Extracting audio track")
            video = VideoFileClip(video_file)
            audio = video.audio
            audio_file = os.path.splitext(video_file)[0] + ".wav"
            audio.write_audiofile(audio_file)
            return audio_file
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None

    # Speech2Text (Whisper)
    def transcribe_audio(self, audio_file, source_language, model_size="tiny"):
        try:
            print("Transcribing audio track")
            model = whisper.load_model(model_size)
            trans = model.transcribe(
                audio_file,
                language=source_language,
                verbose=False,
                word_timestamps=True,
            )
            return trans
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def translate_text(self, texts, src_lang="rus_Cyrl", target_lang="en"):
        try:
            model_name_tr = (
                "Helsinki-NLP/opus-mt-ru-en"  # "facebook/nllb-200-distilled-600M"
            )
            tokenizer_tr = AutoTokenizer.from_pretrained(
                model_name_tr, src_lang=src_lang
            )
            model_tr = AutoModelForSeq2SeqLM.from_pretrained(model_name_tr)
            inputs = tokenizer_tr(texts, return_tensors="pt")
            translated_tokens = model_tr.generate(
                **inputs,
                forced_bos_token_id=tokenizer_tr.convert_tokens_to_ids(target_lang),
            )
            return tokenizer_tr.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
        except Exception as e:
            print(f"Error translating texts: {e}")
            return None

        # src_lang='rus_Cyrl',
        # model_name_tr = "facebook/nllb-200-distilled-600M"
        # tokenizer_tr = AutoTokenizer.from_pretrained(model_name_tr, src_lang=src_lang)
        # model_tr = AutoModelForSeq2SeqLM.from_pretrained(model_name_tr)
        # try:
        #     # Tokenize the list of strings
        #     inputs = tokenizer_tr(texts, return_tensors="pt")

        #     # Generate translations
        #     translated_tokens = model_tr.generate(
        #         **inputs,
        #         forced_bos_token_id=tokenizer_tr.convert_tokens_to_ids(target_lang),
        #         max_length=256
        #     )

        #     # Decode the translated tokens
        #     return self.tokenizer_tr.batch_decode(translated_tokens, skip_special_tokens=True)

        # except Exception as e:
        #     print(f"Error translating texts: {e}")
        #     return None

    # Text2Audio
    def create_audio_from_text(self, text, target_language=None):
        model_name_tts = "facebook/mms-tts-eng"
        model_tts = VitsModel.from_pretrained(model_name_tts)
        tokenizer_tts = AutoTokenizer.from_pretrained(model_name_tts)
        audio_file = "translated_" + str(uuid.uuid4()) + ".wav"
        try:
            inputs = tokenizer_tts(text, return_tensors="pt")
            with torch.no_grad():
                output = model_tts(**inputs).waveform
            output_np = output.squeeze().cpu().numpy()
            # controversial 2 lines below
            with open(audio_file, "wb") as out:
                out.write(output_np)
            scipy.io.wavfile.write(
                audio_file, rate=model_tts.config.sampling_rate, data=output_np
            )
            return audio_file
        except Exception as e:
            if os.path.isfile(audio_file):
                os.remove(audio_file)
            raise Exception(f"Error creating audio from text: {e}")

    def merge_audio_files(
        self, transcription, source_language, target_language, audio_file
    ):
        # source_language = 'russian'
        # target_language = 'english'
        # target_lang = 'en'
        sentences = []
        sentence_ends = []
        sentence = ""
        # sent_start = 0
        print("Composing sentences")
        for segment in tqdm(transcription["segments"]):
            if segment["text"].isupper():
                continue
            for i, word in enumerate(segment["words"]):
                if not self.ISWORD.search(word["word"]):
                    continue
                word["word"] = self.ABBREVIATIONS.get(
                    word["word"].strip(), word["word"]
                )
                if word["word"].startswith("-"):
                    sentence = sentence[:-1] + word["word"] + " "
                else:
                    sentence += word["word"]

                if word["word"].endswith("."):
                    sentences.append(sentence)
                    sentence_ends.append(word["end"])
                    # sent_start = 0
                    sentence = ""

        sentence_starts = [np.float64(0)] + sentence_ends[:-1]

        temp_files = []

        try:
            # Load the original audio
            # ducked_audio = AudioSegment.from_wav(audio_file)

            merged_audio = AudioSegment.silent(duration=0)

            # Translate sentences
            print("Translating sentences...")
            translated_texts = []
            for i in tqdm(range(0, len(sentences), 128)):
                chunk = sentences[i : i + 128]
                translated_chunk = self.translate_text(chunk, target_language)
                if translated_chunk is None:
                    raise Exception("Translation failed")
                translated_texts.extend(translated_chunk)

            # Create translated audio track and align it
            print("Creating translated audio track...")
            for i, translated_text in enumerate(tqdm(translated_texts)):
                translated_audio_file = self.create_audio_from_text(
                    translated_text, target_language
                )
                if translated_audio_file is None:
                    raise Exception("Audio creation failed")
                temp_files.append(translated_audio_file)

                translated_audio = AudioSegment.from_wav(translated_audio_file)

                # Get the start and end times of the sentence
                start_time = sentence_starts[i] * 1000  # convert to milliseconds
                end_time = sentence_ends[i] * 1000  # convert to milliseconds
                original_duration = end_time - start_time

                # Adjust the translated audio length to match the original duration
                if len(translated_audio) < original_duration:
                    translated_audio = self.change_audio_speed(
                        translated_audio, original_duration
                    )

                # Overlay the translated audio over the silent base audio
                padding_before = AudioSegment.silent(
                    duration=start_time - len(merged_audio)
                )
                padding_after = AudioSegment.silent(
                    duration=original_duration - len(translated_audio)
                )
                merged_audio += padding_before + translated_audio + padding_after

            return merged_audio
        except Exception as e:
            print(f"Error merging audio files: {e}")
            return None
        finally:
            # Clean up temporary files
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing temporary file {file}: {e}")

    # Helper function to change the speed of the audio segment (used to stretch/shrink audio)
    def change_audio_speed(self, audio, target_duration):
        original_duration = len(audio)
        if original_duration == 0 or target_duration == 0:
            return audio

        # Ratio of target to original duration
        speed_change_factor = original_duration / target_duration

        # Stretch audio by changing frame rate, making it faster or slower
        altered_audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * speed_change_factor)},
        ).set_frame_rate(audio.frame_rate)

        # Return stretched or compressed audio
        return altered_audio

    def save_audio_to_file(self, audio, filename):
        try:
            audio.export(filename, format="wav")
            print(f"Audio track with translation only saved to {filename}")
        except Exception as e:
            print(f"Error saving audio to file: {e}")

    def replace_audio_in_video(self, video_file, new_audio):
        try:
            # Load the video
            video = VideoFileClip(video_file)

            # Save the new audio to a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_audio_file:
                new_audio.export(temp_audio_file.name, format="wav")
            new_audio.export("duckled.wav", format="wav")

            # Load the new audio into an AudioFileClip
            try:
                new_audio_clip = AudioFileClip(temp_audio_file.name)
            except Exception as e:
                print(f"Error loading new audio into an AudioFileClip: {e}")
                return

            # Check if the audio is compatible with the video
            if new_audio_clip.duration < video.duration:
                print(
                    "Warning: The new audio is shorter than the video. The remaining video will have no sound."
                )
            elif new_audio_clip.duration > video.duration:
                print(
                    "Warning: The new audio is longer than the video. The extra audio will be cut off."
                )
                new_audio_clip = new_audio_clip.subclip(0, video.duration)

            # Set the audio of the video to the new audio
            video = video.set_audio(new_audio_clip)

            # Write the result to a new video file
            output_filename = os.path.splitext(video_file)[0] + "_translated.mp4"
            try:
                video.write_videofile(output_filename, audio_codec="aac")
            except Exception as e:
                print(f"Error writing the new video file: {e}")
                return

            print(f"Translated video saved as {output_filename}")

        except Exception as e:
            print(f"Error replacing audio in video: {e}")
        finally:
            # Remove the temporary audio file
            if os.path.isfile(temp_audio_file.name):
                os.remove(temp_audio_file.name)

    def replace_audio_in_video_and_save(self, video_file, new_audio):
        try:
            # Load the video
            video = VideoFileClip(video_file)

            # Save the new audio to a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_audio_file:
                new_audio.export(temp_audio_file.name, format="wav")
            new_audio.export("duckled.wav", format="wav")

            # Load the new audio into an AudioFileClip
            try:
                # new_audio_clip = AudioFileClip(temp_audio_file.name)
                new_audio_clip = AudioFileClip(new_audio)
            except Exception as e:
                print(f"Error loading new audio into an AudioFileClip: {e}")
                return

            # Check if the audio is compatible with the video
            if new_audio_clip.duration < video.duration:
                print(
                    "Warning: The new audio is shorter than the video. The remaining video will have no sound."
                )
            elif new_audio_clip.duration > video.duration:
                print(
                    "Warning: The new audio is longer than the video. The extra audio will be cut off."
                )
                new_audio_clip = new_audio_clip.subclip(0, video.duration)

            # Set the audio of the video to the new audio
            video = video.set_audio(new_audio_clip)

            # Write the result to a new video file
            output_filename = os.path.splitext(video_file)[0] + "_translated.mp4"
            try:
                video.write_videofile(output_filename, audio_codec="aac")
            except Exception as e:
                print(f"Error writing the new video file: {e}")
                return

            print(f"Translated video saved as {output_filename}")

        except Exception as e:
            print(f"Error replacing audio in video: {e}")
        finally:
            # Remove the temporary audio file
            # if os.path.isfile(temp_audio_file.name):
            #     os.remove(temp_audio_file.name)
            if os.path.isfile(new_audio):
                os.remove(new_audio)
