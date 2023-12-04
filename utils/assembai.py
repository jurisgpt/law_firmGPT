import assemblyai as aai

aai.settings.api_key = "e4aa248aebd940558311227a15b677e6"

audio_url = "/root/workspace/utils/20230607_me_canadian_wildfires.mp3"

transcriber = aai.Transcriber()

transcript = transcriber.transcribe(audio_url)

print("result", transcript.text)