
DISCLAIMER: THIS REPOSITORY IS PROVIDED FOR EDUCATIONAL PURPOSES ONLY. USE AT YOUR OWN RISK. THE AUTHOR IS 
NOT RESPONSIBLE FOR ANY DAMAGE OR LOSS CAUSED BY USING THE CODE. DO NOT USE THIS CODE FOR PRODUCTION ENVIRONMENTS 
OR WITH REAL DATA WITHOUT PROPER VALIDATION AND TESTING.

About This Demo

This demo showcases a simplified simulation of a federated learning system applied to healthcare. Instead of aggregating raw patient data in a central location, each hospital 
trains its own model locally using PII data (I am using Synthetic Data for the demo).

The models send only their learned weights to a central aggregator, which combines them into a global model. This technique protects privacy while enabling collaborative learning — 
a technique increasingly relevant for sensitive domains like healthcare, finance, and government.

The model you are using here was trained using weights from 3 different "hospitals" — simulating real-world data silos. This represents how we can innovate without compromising privacy.

⚠️ This is a conceptual demo built for educational purposes only. The model uses synthetic data and does not reflect actual medical diagnostics.


(c) 2025 The Small Wall Podcast - Srikanth Devarajan. 
