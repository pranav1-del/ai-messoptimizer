import openai

# ⚠️ Add your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"


def generate_menu_report(attendance_data, food_data):
    """
    Uses Generative AI to analyze mess data and generate:
    - Weekly menu
    - Insights for wastage reduction
    - Optimization suggestions
    """

    prompt = f"""
You are an AI-powered college mess manager.

Analyze the following data:

Attendance Data:
{attendance_data}

Food Consumption Data:
{food_data}

Your tasks:
1. Generate a 7-day optimized mess menu.
2. Suggest how to reduce food wastage.
3. Identify patterns between attendance and food consumption.
4. Give simple, practical recommendations.

Format your response clearly with headings:
- Weekly Menu
- Insights
- Recommendations
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst for college mess optimization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]
