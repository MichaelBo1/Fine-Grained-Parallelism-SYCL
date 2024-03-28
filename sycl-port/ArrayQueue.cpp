/*
    - FIFO queue
    - Single producer multi consumer
    - No lock needed
*/
template<typename valueType, int maxSize>
class SPMCArrayQueue
{
    public:
        SPMCArrayQueue() : m_size(0), m_nextElement(0)
        {
        };
        ~SPMCArrayQueue()
        {
        };

        const valueType &front(int i = 0)
        {
            int element = (maxSize + m_nextElement + i - m_size) % maxSize;
            return m_elements[element];
        }
        
        const valueType &back()
        {
            int lastElement = (maxSize + m_nextElement - 1) % maxSize;
            return m_elements[lastElement];
        }

        void push(const valueType &value)
        {
            if (m_size < maxSize)
            {
                m_elements[m_nextElement] = value;
                m_nextElement = (m_nextElement + 1) % maxSize;
                m_size += 1; 
            }
        }

        void pop(int n = 1)
        {
            if (m_size >= n)
            {
                m_size -= n;
            }
        }

        bool empty() const
        {
            return m_size == 0;
        }

        int size() const
        {
            return m_size;
        }

        int sizeMax() const
        {
            return maxSize;
        }

    protected:
        int m_size; // current number of elements in the queue
        int m_nextElement; // index of the next available spot in the element array
        valueType m_elements[maxSize];
};