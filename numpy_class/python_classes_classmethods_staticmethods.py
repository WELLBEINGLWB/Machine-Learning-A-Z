# Python Object-Oriented Programming

class Employee:
      
      raise_factor = 1.04
      num_of_emps = 0
      
      def __init__(self, first, last, pay):
            self.first = first
            self.last = last
            self.pay = pay
            self.email = first + '.' + last + '@company.com'
            Employee.num_of_emps += 1
            
      # class variables are used for all instances
      
      def fullname(self):
            return ('{} {}'.format(self.first, self.last))            
      
      def apply_raise(self):
            self.pay = int(self.pay * self.raise_factor)
#            self.pay = int(self.pay * self.raise_factor)
      
      @classmethod # common convention to call cls class in classmethod
      def set_raise_amt(cls, factor):
            cls.raise_factor = factor
      
      # commonly used when the class is used in a way for classmethod
      @classmethod
      def from_string(cls, emp_str):
            first, last, pay = emp_str.split('-')
            return cls(first, last, pay)
      
      # do not operate on the instances or the class
      @staticmethod
      def is_workday(day):
            if (day.weekday() == 5 or day.weekday() == 6):
                  return False
            return True
      
emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('a', 'b', 5024)

# running class method from class
Employee.set_raise_amt(1.06)
# running class method from object is not good
emp_1.set_raise_amt(2)

print(Employee.raise_factor)
print(emp_1.raise_factor)
print(emp_2.raise_factor)

emp_str_1 = 'JOHN-DOE-3948'
emp_str_2 = 'YEs-Not-12342'
emp_str_3 = 'ABC-DEF-1000'

new_emp_1 = Employee.from_string(emp_str_1)

print(new_emp_1.email)

import datetime
my_date = datetime.date(2016, 7, 11)

print(Employee.is_workday(my_date))
