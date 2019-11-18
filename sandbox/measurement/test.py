import quasar

probabilities = quasar.ProbabilityHistogram(
    nqubit=2,
    histogram={
        1 : 0.1,
        2 : 0.9,
    },  
    nmeasurement=1000,
    # nmeasurement=None,
    )

print(probabilities)
print(probabilities[2])
print(probabilities['10'])

counts = probabilities.to_count_histogram()

print(counts)
print(counts[2])
print(counts['10'])

probabilities2 = counts.to_probability_histogram()
print(probabilities2)
