from typing import TypedDict, NotRequired, Literal, Union

# 1️⃣  Base type with required keys
class Person(TypedDict):
    name: str
    age: int

# 2️⃣  Extend it with optional and literal fields
class Employee(Person):
    # Optional key
    department: NotRequired[str]
    # Literal constrains the allowed values
    role: Literal["Manager", "Developer", "Designer"]

# 3️⃣  Use Union for flexible types
class ContactInfo(TypedDict):
    email: str
    phone: Union[str, int]   # can be string or number

# ✅ Create objects using these types
employee_1: Employee = {
    "name": "Alice",
    "age": 30,
    "role": "Developer",
    "department": "Engineering"   # optional but provided
}

employee_2: Employee = {
    "name": "Bob",
    "age": 45,
    "role": "Manager"             # department omitted
}

contact: ContactInfo = {
    "email": "alice@example.com",
    "phone": "9876543210"
}

print(employee_1)
print(employee_2)
print(contact)
