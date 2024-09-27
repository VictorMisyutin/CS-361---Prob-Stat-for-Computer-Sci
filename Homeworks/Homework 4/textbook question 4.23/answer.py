import numpy as np
import matplotlib.pyplot as plt

def tickets_to_sell(p):
    return np.ceil(10 / p)

def simulate_flight(p, N=10**5):
    seats = 10
    expected_passengers = []
    
    for _ in range(N):
        passengers = np.random.binomial(1, p, size=seats)
        women = np.random.binomial(1, 0.5, size=seats) * passengers
        
        if np.sum(women) >= 2:
            expected_passengers.append(np.sum(passengers))
    
    return np.mean(expected_passengers)

p_values = np.arange(0.1, 1.1, 0.1)
expected_passengers_flying = []

for p in p_values:
    print("tickets to sell if p = ",p ,": ",tickets_to_sell(p))
    
print()

for p in p_values:
    x = simulate_flight(p)
    print("Expected passengersif p = ",p ,": ",x)
    expected_passengers_flying.append(x)
    
plt.plot(p_values, expected_passengers_flying, marker='o')
plt.xlabel('Probability p')
plt.ylabel('Expected Number of Passengers (Flight Flies)')
plt.title('Expected Passengers vs Probability p')
plt.grid(True)
plt.show()
