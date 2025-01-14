
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_sample_data(n_samples=1000):
    np.random.seed(42)

    # Generate credit scores between 300 and 850
    credit_scores = np.clip(np.random.normal(650, 100, n_samples), 300, 850)

    # Calculate default probability based on credit score
    # Higher credit score = lower default probability
    # Using a logistic function to create realistic relationship
    def calculate_default_prob(credit_score):
        # Logistic function that maps credit scores to default probabilities
        # 300 -> ~15% default rate
        # 580 -> ~8% default rate
        # 670 -> ~3% default rate
        # 740 -> ~1% default rate
        # 850 -> ~0.2% default rate
        score_normalized = (credit_score - 300) / (850 - 300)
        default_prob = 0.15 / (1 + np.exp(5 * score_normalized))
        # Add small random noise
        noise = np.random.normal(0, 0.003)
        return np.clip(default_prob + noise, 0, 0.15)

    data = {
        'credit_score': credit_scores,
        'income': np.random.normal(70000, 20000, n_samples),
        'loan_amount': np.clip(np.random.normal(20000, 5000, n_samples), 5000, 50000),
        'debt_to_income': np.clip(np.random.normal(0.3, 0.1, n_samples), 0, 0.6),
        'interest_rate': np.clip(np.random.normal(0.05, 0.02, n_samples), 0.029, 0.15),
        'default_probability': np.array([calculate_default_prob(score) for score in credit_scores])
    }

    return pd.DataFrame(data)


def calculate_optimal_rate(credit_score, income, debt_to_income):
    base_rate = 0.05  # 5% base rate

    # Adjusted credit score impact for the 300-850 range
    credit_score_impact = (700 - credit_score) * 0.0002  # More significant impact
    dti_impact = debt_to_income * 0.1
    income_impact = (50000 - income) / 1000000

    rate = base_rate + credit_score_impact + dti_impact + income_impact
    # Ensure rate stays within reasonable bounds
    return np.clip(rate, 0.029, 0.15)



# Customer segmentation
def segment_customers(df):

    df['segment'] = pd.cut(df['credit_score'],
        bins=[300, 580, 670, 740, 850],
        labels=['Poor', 'Fair', 'Good', 'Excellent']
    )
    return df


class LendingAnalytics:
    def __init__(self, n_samples=1000):
        self.df = generate_sample_data(n_samples)
        self.df = self.process_data()

    def create_risk_profile_visualization(self):
        fig = px.scatter(
            self.df,
            x='credit_score',
            y='default_probability',
            color='segment',
            size='loan_amount',
            hover_data=['income', 'debt_to_income'],
            title='Risk Profile by Customer Segment',
            range_x=[300, 850]  # Set fixed range for credit scores
        )
        return fig

    def create_pricing_optimization_chart(self):
        fig = px.scatter(
            self.df,
            x='credit_score',
            y='optimal_rate',
            color='segment',
            title='Optimized Interest Rates by Credit Score',
            range_x=[300, 850]  # Set fixed range for credit scores
        )

        # Add trendline
        z = np.polyfit(self.df['credit_score'], self.df['optimal_rate'], 1)
        p = np.poly1d(z)

        x_range = np.array([300, 850])  # Use fixed credit score range
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=p(x_range),
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )

        return fig

    def process_data(self):
        # Add calculated fields
        df = segment_customers(self.df)
        df['optimal_rate'] = df.apply(lambda x: calculate_optimal_rate(
            x['credit_score'],
            x['income'],
            x['debt_to_income']
        ), axis=1)

        df['annual_revenue'] = df['loan_amount'] * df['interest_rate'] * (1 - df['default_probability'])  # Expected interest revenue
        df['expected_loss'] = df['loan_amount'] * df['default_probability']  # Principal loss from defaults
        df['expected_profit'] = df['annual_revenue'] - df['expected_loss']

        df['optimal_rate'] = df.apply(lambda x: calculate_optimal_rate(
            x['credit_score'],
            x['income'],
            x['debt_to_income']
        ), axis=1)
        df['rate_difference'] = df['optimal_rate'] - df['interest_rate']

        return df

    def create_profitability_analysis(self):
        # Create a single plot instead of subplots
        fig = go.Figure()

        # Color mapping for segments
        color_map = {
            'Poor': 'red',
            'Fair': 'orange',
            'Good': 'blue',
            'Excellent': 'green'
        }

        # Calculate average profit by segment
        segment_profit = self.df.groupby('segment')[['expected_profit', 'loan_amount']].mean()
        segment_profit['profit_per_1000'] = (segment_profit['expected_profit'] / segment_profit['loan_amount']) * 1000

        fig.add_trace(
            go.Bar(
                x=segment_profit.index,
                y=segment_profit['profit_per_1000'],
                marker_color=list(segment_profit.index.map(color_map)),
                name='Profit per $1000 Loan'
            )
        )

        # Update layout with clear labels
        fig.update_layout(
            title='Average Profit by Customer Segment',
            xaxis_title='Customer Credit Segment',
            yaxis_title='Profit (K)',
            height=500,
            showlegend=False,
            # Add some padding to make labels more visible
            margin=dict(l=80, r=80, t=100, b=80)
        )

        return fig

    def plot_default_rates_by_segment(self):
        fig = go.Figure()

        # Calculate average default rate by segment
        segment_defaults = self.df.groupby('segment')['default_probability'].agg(['mean', 'std'])

        fig.add_trace(go.Bar(
            x=segment_defaults.index,
            y=segment_defaults['mean'] * 100,  # Convert to percentage
            error_y=dict(
                type='data',
                array=segment_defaults['std'] * 100,
                visible=True
            ),
            name='Default Rate'
        ))

        fig.update_layout(
            title='Average Default Rate by Customer Segment',
            yaxis_title='Default Rate (%)',
            xaxis_title='Customer Segment',
            showlegend=True
        )

        return fig


    def generate_segment_insights(self):
        insights = self.df.groupby('segment', observed=True).agg({
            'credit_score': 'mean',
            'income': 'mean',
            'loan_amount': 'mean',
            'default_probability': 'mean',
            'optimal_rate': 'mean',
            'expected_profit': 'mean'
        }).round(2)

        return insights

    def create_dashboard_view(self):
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Profile by Customer Segment',
                'Default Rate by Customer Segment',
                'Optimized Interest Rates by Credit Score',
                'Average Profit by Customer Segment'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Color map for consistency across plots
        color_map = {
            'Poor': 'red',
            'Fair': 'orange',
            'Good': 'blue',
            'Excellent': 'green'
        }

        # 1. Risk Profile (top left)
        fig.add_trace(
            go.Scatter(
                x=self.df['credit_score'],
                y=self.df['default_probability'] * 100,  # Convert to percentage
                mode='markers',
                marker=dict(
                    color=self.df['segment'].map(color_map),
                    size=8
                ),
                name='Risk Profile',
                showlegend=False
            ),
            row=1, col=1
        )

        # Add separate traces for legend
        for segment, color in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(color=color, size=8),
                    name=segment,
                    showlegend=True
                ),
                row=1, col=1
            )

        # 2. Default Rates (top right)
        segment_defaults = self.df.groupby('segment', observed=True)['default_probability'].agg(['mean', 'std'])
        fig.add_trace(
            go.Bar(
                x=segment_defaults.index,
                y=segment_defaults['mean'] * 100,  # Convert to percentage
                marker_color=list(segment_defaults.index.map(color_map)),
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Interest Rates (bottom left)
        fig.add_trace(
            go.Scatter(
                x=self.df['credit_score'],
                y=self.df['optimal_rate'] * 100,  # Convert to percentage
                mode='markers',
                marker=dict(
                    color=self.df['segment'].map(color_map),
                    size=8
                ),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Profitability (bottom right)
        segment_profit = self.df.groupby('segment', observed=True)[['expected_profit', 'loan_amount']].mean()
        segment_profit['profit_per_1000'] = (segment_profit['expected_profit'] / segment_profit['loan_amount']) * 1000

        fig.add_trace(
            go.Bar(
                x=segment_profit.index,
                y=segment_profit['profit_per_1000'],
                marker_color=list(segment_profit.index.map(color_map)),
                showlegend=False,
                text=segment_profit['profit_per_1000'].round(2),  # Add text labels
                textposition='auto',
            ),
            row=2, col=2
        )

        # Update layout and formatting
        fig.update_layout(
            height=900,  # Increase height for better visibility
            width=1200,  # Set width
            title_text="Lending Analytics Dashboard",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                title_text="Credit Score Segments"
            )
        )

        # Update axes labels and formats
        fig.update_xaxes(title_text="Credit Score", row=1, col=1)
        fig.update_yaxes(title_text="Default Probability (%)", row=1, col=1)

        fig.update_xaxes(title_text="Customer Segment", row=1, col=2)
        fig.update_yaxes(title_text="Default Rate (%)", row=1, col=2)

        fig.update_xaxes(title_text="Credit Score", row=2, col=1)
        fig.update_yaxes(title_text="Interest Rate (%)", row=2, col=1)

        fig.update_xaxes(title_text="Customer Segment", row=2, col=2)
        fig.update_yaxes(
            title_text="Profit (K)",
            row=2,
            col=2,
            tickformat='$.2f',  # Format as currency
        )

        # Update y-axis ranges to start at 0
        fig.update_yaxes(rangemode="tozero")

        return fig

    def create_dashboard(self):
        # Create single dashboard view
        dashboard_fig = self.create_dashboard_view()
        dashboard_fig.show()

        # Print insights below the dashboard
        print("\nCustomer Segment Insights:")
        print(self.df.groupby('segment', observed=True)[['expected_profit', 'default_probability', 'interest_rate']].mean())

        print("\nPricing Optimization Opportunities:")
        opportunities = self.df[self.df['rate_difference'] > 0.02]
        print(f"Number of accounts with potential rate increases: {len(opportunities)}")
        print(f"Average potential rate increase: {opportunities['rate_difference'].mean():.2%}")
        print(f"Potential additional revenue: ${opportunities['loan_amount'].sum() * opportunities['rate_difference'].mean():,.2f}")


# Run the analysis
if __name__ == "__main__":
    analytics = LendingAnalytics(n_samples=1000)
    analytics.create_dashboard()