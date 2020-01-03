curl -X POST "https://api.ionq.co/v0/jobs" \
 -H "Authorization: apiKey 7MduR9Pj1ogAdlFFis05S2:5CU6u38eyIj9jZae5IyLHT" \
 -H "Content-Type: application/json" \
 -d '{"lang": "json", "body": {"qubits": 1, "circuit": [{"gate": "h", "target": 0}]}}'
