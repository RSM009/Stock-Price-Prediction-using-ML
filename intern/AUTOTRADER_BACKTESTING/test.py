start = "24/05/2023"  # Replace this with your actual date string

# Extract day, month, and year from the input string
day, month, year = map(int, start.split('/'))

# Create the desired output format
start_dt = f"{year:04d}-{month:02d}-{day:02d} 00:00:00+00:00"

print(start_dt)