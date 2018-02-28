# Python Object-Oriented Programming

class Employee:
      def __init__(self, first, last, pay):
            self.first = first
            self.last = last
            self.pay = pay
            self.email = first + '.' + last + '@company.com'
      
      def fullname(self):
            return ('{} {}'.format(self.first, self.last))            
            
            
emp_1 = Employee('Corey', 'Schafer', 50000)

#emp_1.first = 'Corey' #manually instances can be added
#emp_1.last = 'Schafer'
#emp_1.pay = 50000

#print(emp_1.email)

#print('{} {}'.format(emp1_first, emp_1.last)) # manually printing

print(emp_1.fullname())
# equal lines
print(Employee.fullname(emp_1))
# when classes are generally called 
