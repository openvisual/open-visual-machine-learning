months = "1 2 3 4 5 6 7 8 9 10 11 12 "
s = months.split( " " )
j = "월\n".join( s )
print( j )

months = "1 2 3 4 5 6 7 8 9 10 11 12"
s = months.split( " " )
for m in s :
    print( "%s월" % m )
pass