import gradio as gr
import pandas as pd
import pickle

with open("vgame_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


def predict_sales(rank, name, platform, year, genre, publisher,
                  na_sales, eu_sales, jp_sales, other_sales):
    input_df = pd.DataFrame([[
        rank, name, platform, year, genre, publisher,
        na_sales, eu_sales, jp_sales, other_sales
    ]], columns=[
        'Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher',
        'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'
    ])
    
    prediction = model.predict(input_df)[0]
    return f"Predicted Global Sales: {prediction:.2f} million units"


inputs = [
    gr.Number(label="Rank", value=1),
    gr.Textbox(label="Name", placeholder="Game title"),
    gr.Dropdown(["PS4","PS3","X360","Wii","PC","DS","Switch"], label="Platform"),
    gr.Number(label="Year", value=2010),
    gr.Dropdown(["Action","Sports","Shooter","Role-Playing","Adventure","Misc"], label="Genre"),
    gr.Textbox(label="Publisher", placeholder="Enter publisher name"),
    gr.Number(label="NA Sales (millions)", value=0.0),
    gr.Number(label="EU Sales (millions)", value=0.0),
    gr.Number(label="JP Sales (millions)", value=0.0),
    gr.Number(label="Other Sales (millions)", value=0.0)
]


app = gr.Interface(
    fn=predict_sales,
    inputs=inputs,
    outputs="text",
    title="Video Game Global Sales Predictor",
)

app.launch(share=True)