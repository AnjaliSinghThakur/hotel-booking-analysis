# Hotel Booking Cancellation Analysis

Why do guests cancel hotel bookings, and can we predict it? EDA and prediction on 119,390 bookings from a city hotel and a resort hotel.

## Results
- 119,390 bookings in the raw file, 87,230 after removing duplicates and bookings with no guests
- Cancellation rate: 27.5% after cleaning (37.0% on the raw file)
- Random Forest accuracy 80.2% (F1 59.5%), Logistic Regression accuracy 78.7% (F1 50.3%)
- Lead time is the strongest signal: 8.4% cancel within 7 days of booking vs 39.7% when booked 180+ days ahead
- City Hotel 30.1% vs Resort Hotel 23.5%; Online TA 35.4% vs Direct 14.7%
- Peak season is August (most bookings, highest daily rate); January has the lowest rate

## Method
1. Cleaning: drop duplicates and empty bookings, fill missing values
2. EDA: cancellation by hotel, segment, lead time, deposit type, month
3. Models: Logistic Regression and Random Forest, 80/20 split. `reservation_status` is excluded because it leaks the label.

## Limitations
Two Portuguese hotels, 2015-2017. Non-refundable bookings show 94.7% cancellations, which looks like a data quirk, so it is not used for conclusions.

## Run it
Download `hotel_bookings.csv` (Hotel Booking Demand, Antonio, Almeida and Nunes, 2019) into this folder, then:
```
pip install pandas scikit-learn matplotlib
python hotel_analysis.py
```
Charts are saved in `plots/` and numbers in `results.json`.
