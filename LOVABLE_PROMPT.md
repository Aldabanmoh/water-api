# 🌐 LOVABLE PROMPT — Copy and Paste This Into Lovable

Go to https://lovable.dev → Create New Project → Paste the prompt below.

---

## THE PROMPT (copy everything below this line):

---

Build a modern, responsive agricultural water needs calculator web application. This app connects to a Python ML backend API to predict daily irrigation water requirements for farmers.

## Design & Theme
- Color scheme: earth tones with greens (#1B5E20 primary, #4CAF50 accent, #E8F5E9 backgrounds), warm browns for soil elements, blue for water results
- Modern clean design with card-based layout, subtle shadows, rounded corners
- Include a hero section at the top with the title "Agricultural Water Needs Calculator" and subtitle "AI-powered irrigation predictions based on machine learning"
- Add a small leaf or water drop icon in the header
- Responsive: works on mobile and desktop

## Main Input Form (in a card)

Create a form with these inputs organized in logical groups:

### Group 1: Crop Information
- **Crop Type** — dropdown: wheat, alfalfa, citrus, corn, olive, potato, rice, tomato
- **Growth Stage** — dropdown: development, initial, late_season, mid_season

### Group 2: Weather Conditions
- **Max Temperature (°C)** — number input, default 35, min -5, max 50
- **Min Temperature (°C)** — number input, default 20, min -10, max 40
- **Humidity (%)** — slider from 5 to 100, default 45, show the value
- **Wind Speed (m/s)** — number input, default 2.5, min 0, max 15
- **Solar Radiation (MJ/m²/day)** — number input, default 22, min 2, max 35
- **Rainfall (mm/day)** — number input, default 0, min 0, max 100

### Group 3: Soil & Field
- **Soil Type** — dropdown: clay, clay_loam, loam, sandy, silt
- **Current Soil Moisture (%)** — number input, default 25, min 0, max 50
- **Field Area (hectares)** — number input, default 1.0, min 0.1, max 500

### Group 4: Irrigation
- **Irrigation Method** — dropdown: center_pivot, drip, flood, sprinkler

## Calculate Button
- Large green button labeled "Calculate Water Needs"
- Shows a loading spinner while waiting for API response

## Results Section (appears after calculation)
Show results in a prominent card with:
- **Water Need**: X.XX mm/day (large, bold, green)
- **Total Volume**: XXX,XXX liters (with thousands separator)
- **Volume in m³**: XX.X m³
- **Recommendation**: text from API (in a colored info box)
- Small summary showing: crop, stage, area, irrigation method

## API Integration

The app calls a backend API. Store the API URL in a constant at the top of the code so it's easy to change:

```javascript
const API_URL = "https://YOUR-API-URL.com";
```

When the user clicks Calculate:
1. Collect all form values
2. POST to `${API_URL}/predict` with JSON body containing all inputs
3. Display the response: water_need_mm, volume_liters, volume_m3, recommendation

On page load, call `${API_URL}/info` to get valid dropdown values and model accuracy info. Display the model accuracy somewhere subtle (e.g. footer or tooltip).

## Error Handling
- Show a friendly error message if the API is unreachable
- Validate that temp_min < temp_max before sending
- Show validation errors inline next to the relevant field

## Additional Features
- Add a "Reset" button to clear the form back to defaults
- Add a simple history section below the results that saves the last 5 calculations in memory (not localStorage) showing crop, area, and water need
- Add a footer with "Powered by Machine Learning" and the model accuracy from /info

## Technical Notes
- Use Tailwind CSS for styling
- Use React with TypeScript
- Use fetch for API calls, not axios
- Make the API_URL constant easy to find and change at the top of the file

---

## AFTER LOVABLE GENERATES THE APP:

1. Find the line `const API_URL = "..."` in the code
2. Replace it with your actual Railway/Render URL:
   ```
   const API_URL = "https://water-api-production-XXXX.up.railway.app";
   ```
3. Click "Publish" in Lovable to deploy the website
4. Test it end-to-end!

## OPTIONAL FOLLOW-UP PROMPTS FOR LOVABLE:

After the basic app works, you can ask Lovable to add more features:

- "Add a dark mode toggle"
- "Add Arabic and French language support with a language switcher"
- "Add a chart that shows water need by growth stage for the selected crop"
- "Add the ability to input a 7-day weather forecast and show daily predictions in a table"
- "Add a PDF export button that downloads a report with the inputs and results"
- "Add a map where the user can click their location and auto-fill weather from Open-Meteo API"
