# Time Series Analysis of Brent Oil Prices: Detecting Changes and Associating Causes

This project analyzes Brent oil prices from 1987-2022, detecting structural changes and associating them with major events to provide data-driven insights for the energy industry.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Exploratory Data Analysis EDA](#exploratory-data-analysis-eda)
3. [Statistical and Econometric Models to Refine the Analysis](#statistical-and-econometric-models-to-refine-the-analysis)
4. [Other Potential Factors Influencing Oil Prices](#other-potential-factors-influencing-oil-prices)
5. [Contributing](#contributing)
6. [License](#license)

## Project Objective
The primary objective of this project is to analyze how significant events such as political decisions, conflicts in oil-producing regions, global economic sanctions, and changes in OPEC policies—impact the price of Brent oil. This project will:

1. **Identify Key Events**: Pinpoint the major events over the past decade that have significantly influenced Brent oil prices.
  
2. **Measure Impact**: Assess the degree to which these events contribute to price fluctuations.

3. **Provide Actionable Insights**: Deliver clear, actionable insights that will assist investors, policymakers, and energy companies in understanding and responding to these price changes effectively.

By tackling this issue, Birhan Energies aims to empower its clients to make informed decisions, manage risks more efficiently, and optimize strategies for investment, policy development, and operational planning within the energy sector.

## Exploratory Data Analysis (EDA)

This section provides an in-depth Exploratory Data Analysis (EDA) of the Brent oil price dataset. To access the complete EDA, please click the link below:

[View Full EDA Notebook](https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/notebooks/eda.ipynb)

## Statistical and Econometric Models to Refine the Analysis

#### ARIMA model to the Brent oil prices data

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/model/ARIMAModelResiduals.png" alt="ARIMAplot" />

#### LSTM (Long Short-Term Memory) Model

The training and validation loss plot
<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/model/ltsm_loss_plot.png" alt="ltsm plot" />

The Actual vs Predicted prices plot
<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/model/actual_vs_pridiction_plot.png" alt="ltsm plot"/>

## Other Potential Factors Influencing Oil Price

## Correlation between GDP growth rates of major economies and oil prices

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/corre_btn_gdp_and_oil.png" alt="correlation plot" />

Brent oil prices and GDP growth rates over time
<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/brent_oil_gdp_over_time.png" alt="correlation plot" />

## Correlation between Unemployment rates & Oil consumption patterns

Filtering the data points after 2012 and analyzing the correlation
<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/corr_matrix3_2012.png" alt="correlation plot" width="600"/>

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/umemp_vs_oil_time_2012.png" alt="correlation plot" />

## Analyzing the Effect of currency fluctuations (the USD) , on oil prices

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/corr_matrix4_usd.png" alt="correlation plot" width="600"/>

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/usd_oil_price_time.png" alt="correlation plot" />

## Analyzing growth in renewable energy sources on oil demand and prices.

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/eda/corr_oil_renawable.png" alt="correlation plot" width="600"/>

<img src="https://github.com/Daniel-Andarge/AiML-brent-oil-price-analysis/blob/main/assets/model/reg_line_oil_renawable.png" alt="correlation plot" />

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
