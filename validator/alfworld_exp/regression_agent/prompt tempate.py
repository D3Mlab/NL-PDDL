import openai


def evaluate_predicates(list1, list2, predicate):
    results = []
    
    # Loop over both lists and evaluate the predicate for each pair
    for obj1 in list1:
        for obj2 in list2:
            prompt = f"Evaluate the following predicate between two objects:\n\n" \
                     f"Object 1: {obj1}\n" \
                     f"Object 2: {obj2}\n" \
                     f"Predicate: {predicate}\n\n" \
                     "Please determine if the predicate is true or false and provide reasoning."

            # Call OpenAI's GPT-4 API
            response = openai.Completion.create(
                model="gpt-4o",
                prompt=prompt,
                max_tokens=150,
                temperature=0.2,
                n=1,
                stop=None
            )

            # Extract the result
            result = response.choices[0].text.strip()
            results.append((obj1, obj2, result))
    
    return results

# Example usage:
list1 = ["an apple", "a banana"]
list2 = ["a fruit basket", "a refrigerator"]
predicate = "is contained in"

results = evaluate_predicates(list1, list2, predicate)

# Display results
for obj1, obj2, result in results:
    print(f"Object 1: {obj1}, Object 2: {obj2}, Predicate Result: {result}")
