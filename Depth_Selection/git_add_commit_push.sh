
#!/bin/bash

#take user input for which repository to update and message for update

echo "Input message:"
read var

if [ -z "$var" ]
then
      echo "\$Message cannot be empty!!! git cannot be updated"
else
	git add -A
	git commit -m "$var"
	git push origin master
fi
