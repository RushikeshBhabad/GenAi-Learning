from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# Sample Python code to split
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # grade is a float, e.g., 8.5 or 9.2

    def get_details(self):
        # Return the student's name
        return self.name

    def is_passing(self):
        # Returns True if grade is 6.0 or higher
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

# Perform the split
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
if len(chunks) > 1:
    print("Second chunk:\n", chunks[1])
else:
    print("Only one chunk created.")
