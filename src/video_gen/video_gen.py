import time
from datetime import datetime
from google import genai
from google.genai import types

client = genai.Client()

prompt = """A POV dashcam shot from inside a semi-truck cab driving along a suburban road during the day. \
The camera is mounted at eye level behind the windshield, facing forward with a wide angle. \
A Shell gas station comes into view on the right side of the road — Shell branding with the iconic yellow and red shell logo visible on the canopy, large illuminated price sign clearly visible showing fuel grades and prices. \
The truck slows slightly as it passes, the price sign filling the right portion of the frame. \
Shallow depth of field keeps the price sign sharp against a blurred background. \
Realistic documentary style, natural daylight, engine rumble audible, tires on asphalt. \
The tall pole-mounted Shell price sign has a black background with large replaceable white plastic digit panels. \
Three rows clearly show: 'REGULAR  3.29 9', 'PLUS  3.59 9', 'PREMIUM  3.89 9', \
where the 9 at the end of each price is a smaller superscript representing 9/10 of a cent, \
exactly as seen on real US roadside gas station signs. \
The digits are large, blocky, and evenly spaced — sharp and readable from a moving vehicle. \
The driver says, 'Regular is three twenty-nine today.'"""

operation = client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt=prompt,
)

# Poll the operation status until the video is ready.
while not operation.done:
    print("Waiting for video generation to complete...")
    time.sleep(10)
    operation = client.operations.get(operation)

# Download the generated video.
generated_video = operation.response.generated_videos[0]
client.files.download(file=generated_video.video)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"data/truck_dashcam_{timestamp}.mp4"
generated_video.video.save(output_path)
print(f"Generated video saved to {output_path}")
