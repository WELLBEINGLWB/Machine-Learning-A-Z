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
      
      
      
emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('a', 'b', 5024)

print(emp_1.__dict__) # access instances

print(Employee.__dict__)

print(emp_1.fullname())
# equal lines
print(Employee.fullname(emp_1))
# when classes are generally called 

# class variable can be changed
Employee.raise_factor = 1.05

# class variable can be changed for a single object
emp_1.raise_factor = 1.07


print(Employee.num_of_emps)

# using method within in class
#print(emp_1.pay)
#emp_1.apply_raise()
#print(emp_1.pay)