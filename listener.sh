
PID="68409"
while true
do
    if ps -p $PID > /dev/null
    then
        sleep 5
    else
        break
    fi
    
done

python3 test.py par_256.json