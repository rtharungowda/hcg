for SOME_VAR in $@
do
    git config user.name rtharungowda
    git config user.email rtharun.gowda.cd.ece19@itbhu.ac.in

    git add .
    git commit -m /"$SOME_VAR/"
    git push
done;