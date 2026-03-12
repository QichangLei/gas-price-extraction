Prompt

Extract all visible fuel prices and their corresponding fuel types from gas station sign images. No need to confirm with me, just give me the reutlt files.
I will input 15 images in sequential order (Image 1, Image 2, Image 3, etc.).

For each image and for each price detected, provide:

- Image label (Image 1, Image 2, etc.)
- Fuel type (Regular, Midgrade, Premium, Diesel, etc.)
- Price in U.S. dollars (with precision up to 3 decimal places)
- Gas station brand
- Payment type (e.g., Cash, Credit, NA)
- Confidence level (precise percetage like 92% 95%, as precise as possible)

Text output format (TXT):

Image: [Image number]
Fuel Type: [Fuel type]
Price: $[price with up to 3 decimals]
Gas Station Brand: [brand or NA]
Payment Type: [payment type or NA]
Confidence: [(precise percetage like 92% 95%, as precise as possible)]

CSV output format:

Columns:
Image_Number, Fuel_Type, Price, Gas_Station_Brand, Payment_Type, Confidence

Rules:
- Prices must be reported with up to **3 decimal places**
- If any information cannot be confidently determined, output **NA** for that field
- Do not infer missing information beyond what is visible in the image
